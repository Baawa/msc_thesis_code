import os
import sys
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from model_evaluator import ModelEvaluator
from graph import Graph
from cp_evaluators import ICPEvaluator, ICPWithResamplingEvaluator, create_icp, create_mcp, MCPEvaluator, NodeDegreeMCPEvaluator, create_node_degree_mcp, NodeDegreeWeightedCPEvaluator, create_node_degree_weighted_cp, EmbeddingWeightedCPEvaluator, create_embedding_weighted_cp
import time
from tabulate import tabulate
import matplotlib.pyplot as plt
from lib.data import split, split_dataset
from lib.graphsage import GraphSAGEWithSampling
from lib import evaluation
from lib.logger import Logger
from ogb.nodeproppred import PygNodePropPredDataset
import torch

from torch_geometric.datasets import Reddit
import pandas as pd
from torch_geometric.data import Data


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


def split_reddit_graph(data, timesteps):
    graphs = []

    # timestep 1
    timestep1_indices = torch.nonzero(data.train_mask).reshape(-1).tolist()
    timestep1_data = split(data, timestep1_indices)

    train_data, calibration_indices, test_indices = split_dataset(
        timestep1_data, test_frac=0.2, calibration_frac=0.2)
    graphs.append(Graph(1, timestep1_data, train_data,
                  calibration_indices, test_indices))

    # timestep 2
    timestep2_indices = torch.cat([torch.nonzero(
        data.test_mask), torch.nonzero(data.val_mask)]).reshape(-1).tolist()
    timestep2_data = split(data, timestep2_indices)

    train_data, calibration_indices, test_indices = split_dataset(
        timestep2_data, test_frac=0.2, calibration_frac=0.2)
    graphs.append(Graph(2, timestep2_data, train_data,
                  calibration_indices, test_indices))

    return graphs


def split_bitcoin_graph(data, timesteps):
    time_steps = torch.unique(data.time_steps)

    graphs = []

    for ts in time_steps:
        indices = torch.nonzero(torch.where(data.time_steps == ts, 1, 0))[
            :, 0].tolist()

        ts_data = split(data, indices)

        train_data, calibration_indices, test_indices = split_dataset(
            ts_data, test_frac=0.2, calibration_frac=0.2)
        graphs.append(Graph(ts, ts_data, train_data,
                      calibration_indices, test_indices))

    return graphs


def load_bitcoin_graph():
    # target to torch tensor
    target_df = pd.read_csv(
        "dataset/elliptic_bitcoin_dataset/elliptic_txs_classes.csv")

    # make class binary
    target_df["class"] = target_df["class"].replace("unknown", "-1")
    target_df["class"] = target_df["class"].replace("2", "0")
    target_df["class"] = pd.to_numeric(target_df["class"])
    target = torch.tensor(target_df["class"].values)

    # node features to torch tensor
    x_df = pd.read_csv(
        "dataset/elliptic_bitcoin_dataset/elliptic_txs_features.csv", header=None)
    id_df = x_df[0]
    x_df = x_df.drop(columns=0)  # drop id column
    timestep_df = x_df[1]
    x_df = x_df.drop(columns=1)  # drop timestep column
    x_tensor = torch.tensor(x_df.values).float()

    # replace ids
    id_df = pd.to_numeric(id_df)
    id_df = id_df.reset_index()
    id_df = id_df.rename(columns={"index": "New_ID"})
    id_df = id_df.rename(columns={0: "Old_ID"})
    id_dict = dict(zip(id_df["Old_ID"].values, id_df["New_ID"].values))

    # edges to torch tensor
    edges_df = pd.read_csv(
        "dataset/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv")
    edges_df["txId1"] = pd.to_numeric(edges_df["txId1"])
    edges_df["txId2"] = pd.to_numeric(edges_df["txId2"])

    # replace ids
    edges_df["txId1"] = edges_df["txId1"].apply(lambda x: id_dict[x])
    edges_df["txId2"] = edges_df["txId2"].apply(lambda x: id_dict[x])

    edge_index = torch.LongTensor(
        (edges_df["txId1"].values, edges_df["txId2"].values))

    # timesteps
    timesteps = torch.LongTensor(timestep_df.values)

    # put together graph
    data = Data(x=x_tensor, edge_index=edge_index, y=target)

    data.num_classes = 2
    data.num_features = x_tensor.shape[1]
    data.time_steps = timesteps

    return data


def train_model(graph: Graph, model_args):
    model = GraphSAGEWithSampling(model_args["num_features"], model_args["hidden_dim"],
                      model_args["num_classes"], model_args["num_layers"], [100,10,5]).to(DEVICE)

    # reset the parameters to initial random value
    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=model_args["lr"])

    loss_fn = torch.nn.NLLLoss()

    for epoch in range(1, 1 + model_args["epochs"]):
        print(f"Epoch: {epoch}")
        model.train_model(graph.train_data, optimizer, loss_fn, 100000)

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

    icp_evaluator = ICPEvaluator(
        "arxiv_icp", timesteps, output_dir, CONFIDENCE_LEVEL)
    icp_with_resampling_evaluator = ICPWithResamplingEvaluator(
        "arxiv_icp_with_resampling", timesteps, output_dir, CONFIDENCE_LEVEL)
    mcp_evaluator = MCPEvaluator(
        "arxiv_mcp", timesteps, output_dir, CONFIDENCE_LEVEL)
    nd_mcp_evaluator = NodeDegreeMCPEvaluator(
        "arxiv_node_degree_mcp", timesteps, output_dir, CONFIDENCE_LEVEL)
    nd_weighted_cp_evaluator = NodeDegreeWeightedCPEvaluator(
        "arxiv_node_degree_weighted_cp", timesteps, output_dir, CONFIDENCE_LEVEL)
    embedding_weighted_cp_evaluator = NodeDegreeWeightedCPEvaluator(
        "arxiv_embedding_weighted_cp", timesteps, output_dir, CONFIDENCE_LEVEL)

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

        # capture model performance
        model_evaluator.capture(model, graphs)

        # ICP
        logger.log("running ICP")

        y_hat = model.predict(first_snapshot.data)
        y_hat = y_hat[first_snapshot.calibration_indices]

        y_true = first_snapshot.data.y[first_snapshot.calibration_indices]
        y_true = y_true.reshape(-1).detach()

        icp = create_icp(y_hat, y_true, num_classes)

        icp_evaluator.capture(model, icp, graphs)

        # ICP with resampling
        logger.log("running ICP with resampling")

        icp_with_resampling_evaluator.capture(
            model, graphs, num_classes)

        # MCP
        logger.log("running MCP")

        mcp = create_mcp(model, first_snapshot.data,
                         first_snapshot.calibration_indices)

        mcp_evaluator.capture(model, mcp, graphs)

        # Node degree MCP
        logger.log("running node degree MCP")

        nd_mcp = create_node_degree_mcp(
            model, first_snapshot.data, first_snapshot.calibration_indices, degree_bins)

        nd_mcp_evaluator.capture(model, nd_mcp, graphs, degree_bins)

        # Node degree weighted CP
        logger.log("running node degree weighted CP")

        nd_weighted_cp = create_node_degree_weighted_cp(
            model, first_snapshot.data, first_snapshot.calibration_indices)

        nd_weighted_cp_evaluator.capture(model, nd_weighted_cp, graphs)

        # Embedding weighted CP
        logger.log("running embedding weighted CP")

        embedding_weighted_cp = create_embedding_weighted_cp(
            model, first_snapshot.data, first_snapshot.calibration_indices)

        embedding_weighted_cp_evaluator.capture(
            model, embedding_weighted_cp, graphs)

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

    model_evaluators = []
    icp_evaluators = []
    icp_with_resampling_evaluators = []
    mcp_evaluators = []
    nd_mcp_evaluators = []
    nd_weighted_cp_evaluators = []
    embedding_weighted_cp_evaluators = []
    for i, model in enumerate(models):
        timestep = graphs[i].timestep
        model_evaluator = ModelEvaluator(
            f"graphsage_model_{timestep}", [timestep], output_dir)
        model_evaluators.append(model_evaluator)

        icp_evaluator = ICPEvaluator(
            f"arxiv_icp_{timestep}", [timestep], output_dir, CONFIDENCE_LEVEL)
        icp_evaluators.append(icp_evaluator)

        icp_with_resampling_evaluator = ICPWithResamplingEvaluator(
            f"arxiv_icp_with_resampling_{timestep}", [timestep], output_dir, CONFIDENCE_LEVEL)
        icp_with_resampling_evaluators.append(icp_with_resampling_evaluator)

        mcp_evaluator = MCPEvaluator(
            f"arxiv_mcp_{timestep}", [timestep], output_dir, CONFIDENCE_LEVEL)
        mcp_evaluators.append(mcp_evaluator)

        nd_mcp_evaluator = NodeDegreeMCPEvaluator(
            f"arxiv_node_degree_mcp_{timestep}", [timestep], output_dir, CONFIDENCE_LEVEL)
        nd_mcp_evaluators.append(nd_mcp_evaluator)

        nd_weighted_cp_evaluator = NodeDegreeWeightedCPEvaluator(
            f"arxiv_node_degree_weighted_cp_{timestep}", [timestep], output_dir, CONFIDENCE_LEVEL)
        nd_weighted_cp_evaluators.append(nd_weighted_cp_evaluator)

        embedding_weighted_cp_evaluator = NodeDegreeWeightedCPEvaluator(
            f"arxiv_embedding_weighted_cp_{timestep}", [timestep], output_dir, CONFIDENCE_LEVEL)
        embedding_weighted_cp_evaluators.append(
            embedding_weighted_cp_evaluator)

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
            model_evaluators[i].capture(model, [graph])

            # ICP
            logger.log("running ICP")

            y_hat = model.predict(graph.data)
            y_hat = y_hat[graph.calibration_indices]

            y_true = graph.data.y[graph.calibration_indices]
            y_true = y_true.reshape(-1).detach()

            icp = create_icp(y_hat, y_true, num_classes)

            icp_evaluators[i].capture(model, icp, [graph])

            # ICP with resampling
            logger.log("running ICP with resampling")

            icp_with_resampling_evaluators[i].capture(model, [graph], num_classes)

            # MCP
            logger.log("running MCP")

            mcp = create_mcp(
                model, graph.data, graph.calibration_indices)

            mcp_evaluators[i].capture(model, mcp, [graph])

            # Node degree MCP
            logger.log("running node degree MCP")

            nd_mcp = create_node_degree_mcp(
                model, graph.data, graph.calibration_indices, degree_bins)

            nd_mcp_evaluators[i].capture(model, nd_mcp, [graph], degree_bins)

            # Node degree weighted CP
            logger.log("running node degree weighted CP")

            nd_weighted_cp = create_node_degree_weighted_cp(
                model, graph.data, graph.calibration_indices)

            nd_weighted_cp_evaluators[i].capture(model, nd_weighted_cp, [graph])

            # Embedding weighted CP
            logger.log("running embedding weighted CP")

            embedding_weighted_cp = create_embedding_weighted_cp(
                model, graph.data, graph.calibration_indices)

            embedding_weighted_cp_evaluators[i].capture(model, embedding_weighted_cp, [graph])

    # save evaluators results
    for i, model_evaluator in enumerate(model_evaluators):
        logger.log(f"saving results for model {i}")
        model_evaluator.save_results()
        icp_evaluators[i].save_results()
        icp_with_resampling_evaluators[i].save_results()
        mcp_evaluators[i].save_results()
        nd_mcp_evaluators[i].save_results()
        nd_weighted_cp_evaluators[i].save_results()
        embedding_weighted_cp_evaluators[i].save_results()

    # save graphsage training time
    logger.log("graphsage training times: {}".format(
        graphsage_training_times))
    save_training_time("graphsage_training_time", timesteps,
                       graphsage_training_times, output_dir)


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


def run_reddit():
    output_dir = f"output/reddit/{int(time.time())}/"
    logger = Logger(output_dir)

    logger.log("========REDDIT EXPERIMENT========")

    # download dataset using ogb pytorch geometric loader.
    dataset = Reddit("dataset")

    timesteps = [1, 2]
    num_classes = dataset.num_classes
    data = dataset[0]
    # boundaries[i-1] < input[x] <= boundaries[i]
    degree_bins = torch.tensor([0, 5, 10, 20])
    model_args = {
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
                   model_args, split_reddit_graph, output_dir)
    run_train_every_timestep(
        data, num_classes, timesteps, degree_bins, model_args, split_reddit_graph, output_dir)


def run_bitcoin():
    output_dir = f"output/bitcoin/{int(time.time())}/"
    logger = Logger(output_dir)

    logger.log("========BITCOIN EXPERIMENT========")

    # download dataset using ogb pytorch geometric loader.
    data = load_bitcoin_graph()

    # boundaries[i-1] < input[x] <= boundaries[i]
    degree_bins = torch.tensor([0, 5, 10, 20])
    model_args = {
        "num_layers": 3,
        "hidden_dim": 256,
        "lr": 0.01,  # learning rate
        "epochs": 200,
        "num_classes": data.num_classes,
        "num_features": data.num_features,
    }

    timesteps = torch.unique(data.y)

    logger.log("Config\n\ttimesteps: {}\n\tdegree_bins: {}\n\tmodel_args: {}\n\toutput_dir: {}".format(
        timesteps, degree_bins, model_args, output_dir))

    run_train_once(data, data.num_classes, timesteps, degree_bins,
                   model_args, split_bitcoin_graph, output_dir)
    run_train_every_timestep(
        data, data.num_classes, timesteps, degree_bins, model_args, split_bitcoin_graph, output_dir)


# run experiments
# run_arxiv()
run_reddit()
run_bitcoin()
