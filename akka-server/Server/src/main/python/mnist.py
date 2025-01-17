import logging
import argparse
import sys
import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from pathlib import Path
import json

import syft as sy
from syft.workers import websocket_client
from syft.frameworks.torch.fl import utils

LOG_INTERVAL = 25
logger = logging.getLogger("run_websocket_client")


class LearningMember(object):
    def __init__(self, j):
        self.__dict__ = json.loads(j)


# Loss function
@torch.jit.script
def loss_fn(pred, target):
    return F.nll_loss(input=pred, target=target)


# Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def define_and_get_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Run federated learning using websocket client workers."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of the training")
    parser.add_argument(
        "--test_batch_size", type=int, default=128, help="batch size used for the test data"
    )
    parser.add_argument(
        "--training_rounds", type=int, default=40, help="number of federated learning rounds"
    )
    parser.add_argument(
        "--federate_after_n_batches",
        type=int,
        default=10,
        help="number of training steps performed on each remote worker before averaging",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--seed", type=int, default=1, help="seed used for randomization")
    parser.add_argument("--save_model", action="store_true", help="if set, model will be saved")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket client workers will be started in verbose mode",
    )

    parser.add_argument("--datapath", help="show program version", action="store", default="../data")
    parser.add_argument("--participantsjsonlist", help="show program version", action="store", default="{}")
    parser.add_argument("--epochs", type=int, help="show program version", action="store", default=10)
    parser.add_argument("--modelpath")

    args = parser.parse_args(args=args)
    return args


async def fit_model_on_worker(
    worker: websocket_client.WebsocketClientWorker,
    traced_model: torch.jit.ScriptModule,
    batch_size: int,
    curr_round: int,
    max_nr_batches: int,
    lr: float,
):
    """Send the model to the worker and fit the model on the worker's training data.
    Args:
        worker: Remote location, where the model shall be trained.
        traced_model: Model which shall be trained.
        batch_size: Batch size of each training step.
        curr_round: Index of the current training round (for logging purposes).
        max_nr_batches: If > 0, training on worker will stop at min(max_nr_batches, nr_available_batches).
        lr: Learning rate of each training step.
    Returns:
        A tuple containing:
            * worker_id: Union[int, str], id of the worker.
            * improved model: torch.jit.ScriptModule, model after training at the worker.
            * loss: Loss on last training batch, torch.tensor.
    """
    train_config = sy.TrainConfig(
        model=traced_model,
        loss_fn=loss_fn,
        batch_size=batch_size,
        shuffle=True,
        max_nr_batches=max_nr_batches,
        epochs=1,
        optimizer="SGD",
        optimizer_args={"lr": lr},
    )
    train_config.send(worker)
    loss = await worker.async_fit(dataset_key="mnist", return_ids=[0])
    model = train_config.model_ptr.get().obj
    return worker.id, model, loss


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item() # sum up batch loss
            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )

async def main():
    args = define_and_get_arguments()

    hook = sy.TorchHook(torch)

    kwargs_websocket = {"hook": hook, "verbose": args.verbose, "host": "localhost"}

    test_loader = torch.utils.data.DataLoader(datasets.MNIST(args.datapath, train=False, download=True,
                                                             transform=transforms.Compose([
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize(
                                                                     (0.1307,), (0.3081,))
                                                             ])), batch_size=1000, shuffle=True)
    
    print(args.participantsjsonlist)
    participants = args.participantsjsonlist.replace("'","\"")
    print(participants)
    participants = json.loads(participants)
    print(participants)

    #for p in participants:
       # print(f"id: {p['id']}, port: {p['port']}")

    worker_instances = []
    for participant in participants:
        print("----------------------")
        print(participant['id'])
        print(participant['port'])
        print("----------------------")
        worker_instances.append(sy.workers.websocket_client.WebsocketClientWorker(id=participant['id'], port=participant['port'], **kwargs_websocket))

    for wcw in worker_instances:
        wcw.clear_objects_remote()

    use_cuda = args.cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    model_file = Path(args.modelpath)
    if model_file.is_file():
        model = Net().to(device)
        model.load_state_dict(torch.load(args.modelpath))

        test(model, test_loader)
        traced_model = torch.jit.trace(model, torch.zeros([1, 1, 28, 28], dtype=torch.float).to(device))
        # traced_model.eval()
    else:
        model = Net().to(device)
        traced_model = torch.jit.trace(model, torch.zeros([1, 1, 28, 28], dtype=torch.float).to(device))



    learning_rate = args.lr


    for curr_round in range(1, args.epochs + 1):
        logger.info("Training epoch %s/%s", curr_round, args.epochs)

        results = await asyncio.gather(
            *[
                fit_model_on_worker(
                    worker=worker,
                    traced_model=traced_model,
                    batch_size=args.batch_size,
                    curr_round=curr_round,
                    max_nr_batches=args.federate_after_n_batches,
                    lr=learning_rate,
                )
                for worker in worker_instances
            ]
        )
        models = {}
        loss_values = {}

        test_models = curr_round % 10 == 1 or curr_round == args.epochs or curr_round == 0

        # Federate models (note that this will also change the model in models[0]
        for worker_id, worker_model, worker_loss in results:
            if worker_model is not None:
                models[worker_id] = worker_model
                loss_values[worker_id] = worker_loss

        traced_model = utils.federated_avg(models)

        if test_models:
            test(traced_model, test_loader)

        # decay learning rate
        learning_rate = max(0.98 * learning_rate, args.lr * 0.01)

    if args.modelpath:
        torch.save(traced_model.state_dict(), args.modelpath)


if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.DEBUG)

    # Websockets setup
    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.INFO)
    websockets_logger.addHandler(logging.StreamHandler())

    # Run main
    asyncio.get_event_loop().run_until_complete(main())