import os
import sys
# module_path = os.path.abspath(os.path.join('../'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

from model_evaluator import ModelEvaluator
from graph import Graph
from cp_evaluators import ICPEvaluator, ICPWithResamplingEvaluator, create_icp, create_mcp, MCPEvaluator, NodeDegreeMCPEvaluator, create_node_degree_mcp, NodeDegreeWeightedCPEvaluator, create_node_degree_weighted_cp, EmbeddingWeightedCPEvaluator, create_embedding_weighted_cp
import time
from tabulate import tabulate
import matplotlib.pyplot as plt
from data import split, split_dataset
from graphsage import GraphSAGEWithSampling, GraphSAGE
import evaluation
from logger import Logger
from ogb.nodeproppred import PygNodePropPredDataset
import torch


# special setting for plotting on ubuntu
# create virtual display with size 1600x1200 and 16 bit color. Color can be changed to 24 or 8
os.system('Xvfb :1 -screen 0 1600x1200x16  &')
# tell X clients to use our virtual DISPLAY :1.0.
os.environ['DISPLAY'] = ':1.0'


NUM_EXPERIMENTS = 5
CONFIDENCE_LEVEL = 0.95
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def split_arxiv_graph(data, years):
    graphs = []

    for year in years:
        indices = torch.nonzero(torch.where(data.node_year[:, 0] <= year, 1, 0))[
            :, 0].tolist()

        year_data = split(data, indices)

        train_data, calibration_indices, test_indices = split_dataset(
            year_data, test_frac=0.2, calibration_frac=0.2)
        graphs.append(Graph(year, year_data, train_data,
                      calibration_indices, test_indices))

    return graphs

def train_model(graph: Graph, model_args):
    if model_args["use_sampling"] == True:
        model = GraphSAGEWithSampling(model_args["num_features"], model_args["hidden_dim"], model_args["num_classes"], model_args["num_layers"], model_args["sampling_size"], model_args["batch_size"]).to(DEVICE)
    else:
        model = GraphSAGE(model_args["num_features"], model_args["hidden_dim"], model_args["num_classes"], model_args["num_layers"]).to(DEVICE)

    # reset the parameters to initial random value
    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=model_args["lr"])

    loss_fn = torch.nn.NLLLoss()

    for epoch in range(1, 1 + model_args["epochs"]):
        model.train_model(graph.train_data, optimizer, loss_fn)
    
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


def save_training_time(prefix, timesteps, times, output_dir):
    save_results(output_dir, "{}_time.txt".format(
        prefix), tabulate([times], headers=timesteps))


def plot(title, x_label, y_label, x, y, output_dir):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, y, "+-")
    plt.savefig(output_dir + title + ".png")
    plt.close()


def run_train_once(data, num_classes, timesteps, degree_bins, model_args, split_graph, output_dir):
    output_dir = output_dir + "run_train_once/"
    logger = Logger(output_dir)

    logger.log("STARTED: run_train_once")

    plot_class_distribution(
        "full graph", data.y.reshape(-1).detach().numpy(), num_classes, output_dir)

    logger.log('Device: {}'.format(DEVICE))

    # split graph
    graphs = split_graph(data, timesteps)

    for graph in graphs:
        plot_class_distribution(
            graph.timestep, graph.data.y.reshape(-1).detach().numpy(), num_classes, output_dir)
    for graph in graphs:
        graph.train_data = graph.train_data.to(DEVICE)
        graph.data = graph.data.to(DEVICE)

    # train on first snapshot
    first_snapshot = graphs[0]

    start_time = time.time()

    model = train_model(first_snapshot, model_args)

    graphsage_training_time = time.time() - start_time

    y_hat = model.predict(first_snapshot.data)
    y_hat = y_hat[first_snapshot.test_indices]
    y_hat = y_hat.argmax(dim=-1, keepdim=True).reshape(-1)

    y_true = first_snapshot.data.y[first_snapshot.test_indices
                                   ].reshape(-1)

    acc, macro_f1 = evaluation.get_multiclass_classification_performance(
        y_hat.detach().cpu(), y_true.detach().cpu())

    logger.log(
        f"Finished training GraphSAGE model with accuracy {100 * acc:.2f}% and macro avg f1 score: {100 * macro_f1:.2f}%")

    model_evaluator = ModelEvaluator("graphsage_model", timesteps, output_dir)

    icp_evaluator = ICPEvaluator("icp", timesteps, output_dir, CONFIDENCE_LEVEL)
    icp_with_resampling_evaluator = ICPWithResamplingEvaluator("icp_with_resampling", timesteps, output_dir, CONFIDENCE_LEVEL)
    mcp_evaluator = MCPEvaluator("mcp", timesteps, output_dir, CONFIDENCE_LEVEL)
    nd_mcp_evaluator = NodeDegreeMCPEvaluator("node_degree_mcp", timesteps, output_dir, CONFIDENCE_LEVEL)
    nd_weighted_cp_evaluator = NodeDegreeWeightedCPEvaluator("node_degree_weighted_cp", timesteps, output_dir, CONFIDENCE_LEVEL)
    embedding_weighted_cp_evaluator = NodeDegreeWeightedCPEvaluator("embedding_weighted_cp", timesteps, output_dir, CONFIDENCE_LEVEL)

    for experiment_num in range(NUM_EXPERIMENTS):
        logger.log("Experiment {} started".format(experiment_num))

        # split graph
        graphs = split_graph(data, timesteps)

        for graph in graphs:
            plot_class_distribution(
                graph.timestep, graph.data.y.reshape(-1).detach().numpy(), num_classes, output_dir)
        for graph in graphs:
            graph.train_data = graph.train_data.to(DEVICE)
            graph.data = graph.data.to(DEVICE)

        first_snapshot = graphs[0]

        for graph in graphs:
            # capture model performance
            model_evaluator.capture(model, graph)

            # ICP
            logger.log("running ICP")

            y_hat = model.predict(first_snapshot.data)
            y_hat = y_hat[first_snapshot.calibration_indices]

            y_true = first_snapshot.data.y[first_snapshot.calibration_indices]
            y_true = y_true.reshape(-1).detach()

            icp = create_icp(y_hat, y_true, num_classes)

            icp_evaluator.capture(model, icp, graph)

            # ICP with resampling
            logger.log("running ICP with resampling")

            icp_with_resampling_evaluator.capture(model, graph, num_classes)

            # MCP
            logger.log("running MCP")

            mcp = create_mcp(model, first_snapshot.data, first_snapshot.calibration_indices)

            mcp_evaluator.capture(model, mcp, graph)

            # Node degree MCP
            logger.log("running node degree MCP")

            nd_mcp = create_node_degree_mcp(model, first_snapshot.data, first_snapshot.calibration_indices, degree_bins)

            nd_mcp_evaluator.capture(model, nd_mcp, graph, degree_bins)

            # Node degree weighted CP
            logger.log("running node degree weighted CP")

            nd_weighted_cp = create_node_degree_weighted_cp(model, first_snapshot.data, first_snapshot.calibration_indices)

            nd_weighted_cp_evaluator.capture(model, nd_weighted_cp, graph)

            # Embedding weighted CP
            logger.log("running embedding weighted CP")

            embedding_weighted_cp = create_embedding_weighted_cp(model, first_snapshot.data, first_snapshot.calibration_indices)

            embedding_weighted_cp_evaluator.capture(model, embedding_weighted_cp, graph)
        
        model_evaluator.new_batch()
        icp_evaluator.new_batch()
        icp_with_resampling_evaluator.new_batch()
        mcp_evaluator.new_batch()
        nd_mcp_evaluator.new_batch()
        nd_weighted_cp_evaluator.new_batch()
        embedding_weighted_cp_evaluator.new_batch()

    # save graphsage training time
    save_training_time("graphsage_training", timesteps, [
                       graphsage_training_time], output_dir)

    # plot model performance
    model_evaluator.save_results()

    # print cp performance
    icp_evaluator.save_results()
    icp_with_resampling_evaluator.save_results()
    mcp_evaluator.save_results()
    nd_mcp_evaluator.save_results()
    nd_weighted_cp_evaluator.save_results()
    embedding_weighted_cp_evaluator.save_results()


def run_train_every_timestep(data, num_classes, timesteps, degree_bins, model_args, split_graph, output_dir):
    output_dir = output_dir + "run_train_every_timestep/"
    logger = Logger(output_dir)

    logger.log("STARTED: run_train_every_timestep")

    plot_class_distribution(
        "full graph", data.y.reshape(-1).detach().numpy(), num_classes, output_dir)

    logger.log('Device: {}'.format(DEVICE))

    # split graph
    graphs = split_graph(data, timesteps)

    for graph in graphs:
        plot_class_distribution(
            graph.timestep, graph.data.y.reshape(-1).detach().numpy(), num_classes, output_dir)
    for graph in graphs:
        graph.train_data = graph.train_data.to(DEVICE)
        graph.data = graph.data.to(DEVICE)

    graphsage_training_times = []
    models = []
    for graph in graphs:
        start_time = time.time()

        model = train_model(graph, model_args)

        graphsage_training_time = time.time() - start_time
        graphsage_training_times.append(graphsage_training_time)

        models.append(model)

        y_hat = model.predict(graph.data)
        y_hat = y_hat[graph.test_indices]
        y_hat = y_hat.argmax(dim=-1, keepdim=True).reshape(-1)

        y_true = graph.data.y[graph.test_indices].reshape(-1)

        acc, macro_f1 = evaluation.get_multiclass_classification_performance(
            y_hat.detach().cpu(), y_true.detach().cpu())

        logger.log(
            f"Finished training GraphSAGE model with accuracy {100 * acc:.2f}% and macro avg f1 score: {100 * macro_f1:.2f}% for timestep: {graph.timestep}")
    
    model_evaluator = ModelEvaluator("graphsage_model", timesteps, output_dir)

    icp_evaluator = ICPEvaluator("icp", timesteps, output_dir, CONFIDENCE_LEVEL)
    icp_with_resampling_evaluator = ICPWithResamplingEvaluator("icp_with_resampling", timesteps, output_dir, CONFIDENCE_LEVEL)
    mcp_evaluator = MCPEvaluator("mcp", timesteps, output_dir, CONFIDENCE_LEVEL)
    nd_mcp_evaluator = NodeDegreeMCPEvaluator("node_degree_mcp", timesteps, output_dir, CONFIDENCE_LEVEL)
    nd_weighted_cp_evaluator = NodeDegreeWeightedCPEvaluator("node_degree_weighted_cp", timesteps, output_dir, CONFIDENCE_LEVEL)
    embedding_weighted_cp_evaluator = NodeDegreeWeightedCPEvaluator("embedding_weighted_cp", timesteps, output_dir, CONFIDENCE_LEVEL)

    for experiment_num in range(NUM_EXPERIMENTS):
        logger.log("Experiment {} started".format(experiment_num))

        # split graph
        graphs = split_graph(data, timesteps)

        for graph in graphs:
            plot_class_distribution(
                graph.timestep, graph.data.y.reshape(-1).detach().numpy(), num_classes, output_dir)
        for graph in graphs:
            graph.train_data = graph.train_data.to(DEVICE)
            graph.data = graph.data.to(DEVICE)

        for i, model in enumerate(models):
            graph = graphs[i]
            
            # capture model performance
            model_evaluator.capture(model, graph)

            # ICP
            logger.log("running ICP")

            y_hat = model.predict(graph.data)
            y_hat = y_hat[graph.calibration_indices]

            y_true = graph.data.y[graph.calibration_indices]
            y_true = y_true.reshape(-1).detach()

            icp = create_icp(y_hat, y_true, num_classes)

            icp_evaluator.capture(model, icp, graph)

            # ICP with resampling
            logger.log("running ICP with resampling")

            icp_with_resampling_evaluator.capture(model, graph, num_classes)

            # MCP
            logger.log("running MCP")

            mcp = create_mcp(
                model, graph.data, graph.calibration_indices)

            mcp_evaluator.capture(model, mcp, graph)

            # Node degree MCP
            logger.log("running node degree MCP")

            nd_mcp = create_node_degree_mcp(
                model, graph.data, graph.calibration_indices, degree_bins)

            nd_mcp_evaluator.capture(model, nd_mcp, graph, degree_bins)

            # Node degree weighted CP
            logger.log("running node degree weighted CP")

            nd_weighted_cp = create_node_degree_weighted_cp(
                model, graph.data, graph.calibration_indices)

            nd_weighted_cp_evaluator.capture(model, nd_weighted_cp, graph)

            # Embedding weighted CP
            logger.log("running embedding weighted CP")

            embedding_weighted_cp = create_embedding_weighted_cp(
                model, graph.data, graph.calibration_indices)

            embedding_weighted_cp_evaluator.capture(model, embedding_weighted_cp, graph)
        
        model_evaluator.new_batch()
        icp_evaluator.new_batch()
        icp_with_resampling_evaluator.new_batch()
        mcp_evaluator.new_batch()
        nd_mcp_evaluator.new_batch()
        nd_weighted_cp_evaluator.new_batch()
        embedding_weighted_cp_evaluator.new_batch()

    # save evaluators results
    model_evaluator.save_results()
    icp_evaluator.save_results()
    icp_with_resampling_evaluator.save_results()
    mcp_evaluator.save_results()
    nd_mcp_evaluator.save_results()
    nd_weighted_cp_evaluator.save_results()
    embedding_weighted_cp_evaluator.save_results()

    # save graphsage training time
    logger.log("graphsage training times: {}".format(graphsage_training_times))
    save_training_time("graphsage_training_time", timesteps, graphsage_training_times, output_dir)


def run_arxiv():
    output_dir = f"output/arxiv/{int(time.time())}/"
    logger = Logger(output_dir)

    logger.log("========ARXIV EXPERIMENT========")

    # download dataset using ogb pytorch geometric loader.
    dataset = PygNodePropPredDataset(name="ogbn-arxiv")

    # arxiv specific
    timesteps = [2010, 2011, 2012, 2013, 2014,
                 2015, 2016, 2017, 2018, 2019, 2020]
    num_classes = dataset.num_classes
    data = dataset[0]
    # boundaries[i-1] < input[x] <= boundaries[i]
    degree_bins = torch.tensor([0, 5, 10, 20])
    model_args = {
        "use_sampling": False,
        "num_layers": 3,
        "hidden_dim": 256,
        "lr": 0.01,  # learning rate
        "epochs": 200,
        "num_classes": num_classes,
        "num_features": data.num_features,
    }

    logger.log("Config\n\ttimesteps: {}\n\tdegree_bins: {}\n\tmodel_args: {}\n\toutput_dir: {}".format(
        timesteps, degree_bins, model_args, output_dir))

    run_train_once(data, num_classes, timesteps, degree_bins,
                   model_args, split_arxiv_graph, output_dir)
    run_train_every_timestep(
        data, num_classes, timesteps, degree_bins, model_args, split_arxiv_graph, output_dir)


# run experiments
run_arxiv()
