import os
import sys
import parsl
from parsl.app.app import python_app, bash_app
from parsl.config import Config
from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from channels import SSHIteractiveChannel
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_hostname

#woker job
@python_app(executors=["master_htex"])
def runANN(inputs=[]):
    import time
    import datetime
    import socket

    time.sleep(5)
    ts = datetime.datetime.now()
    h = socket.gethostname()

    return "Run an ANN! [" + h + "] > " + inputs[0] + " finished time: " + str(ts)

#woker job
@python_app(executors=["tauri_htex"])
def runBNN(inputs=[]):
    import time
    import datetime
    import socket

    time.sleep(2)
    ts = datetime.datetime.now()
    h = socket.gethostname()

    return "Run a BNN! [" + h + "] > " + inputs[0] + " finished time: " + str(ts)


#master job
def run():
    hostname = sys.argv[1]
    port = sys.argv[2]
    username = sys.argv[3]
    passwd = sys.argv[4]

    os.environ['SSH_AUTH_SOCK'] = '/tmp/ssh-QDMRbSPyH97X/agent.867'

    parsl.load(executionConfig(hostname, port, username, passwd))
    print("... config loaded!")

    data = ['dataset_f']
    ann = runANN(data)
    bnn = runBNN(data)

    outputs = [i.result() for i in [ann, bnn]]

    print(outputs)

def executionConfig(hostname, port, username, passwd):
    config = Config(
        executors=[
            HighThroughputExecutor(
                label="tauri_htex",
                worker_debug=True,
                worker_port_range=(48000, 55000),
                address=address_by_hostname()
            ),
            HighThroughputExecutor(
                label="master_htex",
                worker_debug=True,
                cores_per_worker=1,
                provider=LocalProvider(
                    channel=LocalChannel()
                ),
            )
        ])

    ssh_config = SSHIteractiveChannel(hostname=hostname, username=username, password=passwd, port=port)
    config.executors[0].provider.channel = ssh_config

    return config


if __name__ == '__main__':
    print("> Run ...")
    run()
    print("> Done!")
