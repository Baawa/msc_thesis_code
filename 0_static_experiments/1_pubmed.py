import os

from model_evaluator import ModelEvaluator
from graph import Graph
from cp_evaluators import ICPEvaluator, create_icp, create_mcp, MCPEvaluator, NodeDegreeMCPEvaluator, create_node_degree_mcp, NodeDegreeWeightedCPEvaluator, create_node_degree_weighted_cp, EmbeddingWeightedCPEvaluator, create_embedding_weighted_cp
import time
from tabulate import tabulate
import matplotlib.pyplot as plt
from data import split_dataset
from graphsage import GraphSAGEWithSampling, GraphSAGE
import evaluation
from logger import Logger
from torch_geometric.datasets import Planetoid
import torch


# special setting for plotting on ubuntu
# create virtual display with size 1600x1200 and 16 bit color. Color can be changed to 24 or 8
os.system('Xvfb :1 -screen 0 1600x1200x16  &')
# tell X clients to use our virtual DISPLAY :1.0.
os.environ['DISPLAY'] = ':1.0'


NUM_EXPERIMENTS = 20
CONFIDENCE_LEVEL = 0.95
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(train_data, model_args):
    if model_args["use_sampling"] == True:
        model = GraphSAGEWithSampling(model_args["num_features"], model_args["hidden_dim"], model_args["num_classes"], model_args["num_layers"], model_args["sampling_size"], model_args["batch_size"]).to(DEVICE)
    else:
        model = GraphSAGE(model_args["num_features"], model_args["hidden_dim"], model_args["num_classes"], model_args["num_layers"]).to(DEVICE)

    # reset the parameters to initial random value
    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=model_args["lr"])

    loss_fn = torch.nn.NLLLoss()

    for epoch in range(1, 1 + model_args["epochs"]):
        model.train_model(train_data, optimizer, loss_fn)
    
    return model

def save_results(output_dir, file, str):
    try:
        os.makedirs(output_dir, exist_ok=True)
    finally:
        f = open(output_dir + file, "w")
        f.write(str)
        f.close()


def plot_class_distribution(ext, y, num_classes, output_dir):
    plt.title("Class distribution {}".format(ext))
    plt.hist(y, num_classes)
    plt.xlabel("Class")
    plt.ylabel("num of nodes")
    plt.savefig(output_dir + "class-dist-{}.png".format(ext))
    plt.close()


def save_training_time(prefix, times, output_dir):
    save_results(output_dir, "{}_time.txt".format(prefix), tabulate([times], tablefmt="tsv"))


def run_static_experiment(data, num_classes, degree_bins, model_args, output_dir):
    output_dir = output_dir + "run_static_experiment/"
    logger = Logger(output_dir)

    logger.log("STARTED: run_static_experiment")

    plot_class_distribution("full graph", data.y.reshape(-1).detach().numpy(), num_classes, output_dir)

    logger.log('Device: {}'.format(DEVICE))

    # split data set
    train_data, _, test_indices = split_dataset(data, test_frac=0.2, calibration_frac=0.2)
    
    data = data.to(DEVICE)
    train_data = train_data.to(DEVICE)

    start_time = time.time()

    model = train_model(train_data, model_args)

    graphsage_training_time = time.time() - start_time

    y_hat = model.predict(data)
    y_hat = y_hat[test_indices]
    y_hat = y_hat.argmax(dim=-1, keepdim=True).reshape(-1)

    y_true = data.y[test_indices].reshape(-1)

    acc, macro_f1 = evaluation.get_multiclass_classification_performance(y_hat.detach().cpu(), y_true.detach().cpu())

    logger.log(
        f"Finished training GraphSAGE model with accuracy {100 * acc:.2f}% and macro avg f1 score: {100 * macro_f1:.2f}%")

    model_evaluator = ModelEvaluator("graphsage_model", [1], output_dir)

    icp_evaluator = ICPEvaluator("icp", [1], output_dir, CONFIDENCE_LEVEL)
    mcp_evaluator = MCPEvaluator("mcp", [1], output_dir, CONFIDENCE_LEVEL)
    nd_mcp_evaluator = NodeDegreeMCPEvaluator("node_degree_mcp", [1], output_dir, CONFIDENCE_LEVEL)
    nd_weighted_cp_evaluator = NodeDegreeWeightedCPEvaluator("node_degree_weighted_cp", [1], output_dir, CONFIDENCE_LEVEL)
    embedding_weighted_cp_evaluator = EmbeddingWeightedCPEvaluator("embedding_weighted_cp", [1], output_dir, CONFIDENCE_LEVEL)

    for experiment_num in range(NUM_EXPERIMENTS):
        logger.log("Experiment {} started".format(experiment_num))

        # split data set
        _, calibration_indices, test_indices = split_dataset(data.cpu(), test_frac=0.2, calibration_frac=0.2)

        print(f"data is cuda: {data.is_cuda}")
        graph = Graph(1, data.to(DEVICE), train_data, calibration_indices, test_indices)
        
        # capture model performance
        model_evaluator.capture(model, graph)

        # ICP
        logger.log("running ICP")

        y_hat = model.predict(data)
        y_hat = y_hat[calibration_indices]

        y_true = data.y[calibration_indices]
        y_true = y_true.reshape(-1).detach()

        icp = create_icp(y_hat, y_true, num_classes)

        icp_evaluator.capture(model, icp, graph)

        # MCP
        logger.log("running MCP")

        mcp = create_mcp(model, data, calibration_indices)

        mcp_evaluator.capture(model, mcp, graph)

        # Node degree MCP
        logger.log("running node degree MCP")

        nd_mcp = create_node_degree_mcp(model, data, calibration_indices, degree_bins)

        nd_mcp_evaluator.capture(model, nd_mcp, graph, degree_bins)

        # Node degree weighted CP
        logger.log("running node degree weighted CP")

        nd_weighted_cp = create_node_degree_weighted_cp(model, data, calibration_indices)

        nd_weighted_cp_evaluator.capture(model, nd_weighted_cp, graph)

        # Embedding weighted CP
        logger.log("running embedding weighted CP")

        embedding_weighted_cp = create_embedding_weighted_cp(model, data, calibration_indices)

        embedding_weighted_cp_evaluator.capture(model, embedding_weighted_cp, graph)
        
        model_evaluator.new_batch()
        icp_evaluator.new_batch()
        mcp_evaluator.new_batch()
        nd_mcp_evaluator.new_batch()
        nd_weighted_cp_evaluator.new_batch()
        embedding_weighted_cp_evaluator.new_batch()

    # save graphsage training time
    save_training_time("graphsage_training", [graphsage_training_time], output_dir)

    # plot model performance
    model_evaluator.save_results()

    # print cp performance
    icp_evaluator.save_results()
    mcp_evaluator.save_results()
    nd_mcp_evaluator.save_results()
    nd_weighted_cp_evaluator.save_results()
    embedding_weighted_cp_evaluator.save_results()


def run_arxiv():
    output_dir = f"output/pubmed/{int(time.time())}/"
    logger = Logger(output_dir)

    logger.log("========PUBMED EXPERIMENT========")

    # download dataset using ogb pytorch geometric loader.
    dataset = Planetoid("dataset", "PubMed")

    # arxiv specific
    num_classes = dataset.num_classes
    data = dataset[0]
    # boundaries[i-1] < input[x] <= boundaries[i]
    degree_bins = torch.tensor([0, 1, 5, 10])
    model_args = {
        "use_sampling": False,
        "num_layers": 2,
        "hidden_dim": 256,
        "lr": 0.01,  # learning rate
        "epochs": 100,
        "num_classes": num_classes,
        "num_features": data.num_features,
    }

    logger.log("Config\n\tdegree_bins: {}\n\tmodel_args: {}\n\toutput_dir: {}".format(degree_bins, model_args, output_dir))
    
    run_static_experiment(data, num_classes, degree_bins, model_args, output_dir)


# run experiments
run_arxiv()
