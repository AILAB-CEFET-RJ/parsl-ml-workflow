import sys
import parsl
import pickle
import datetime
import numpy as np

from modules.sampling.data_util import build_dataset

from parsl.app.app import python_app
from parsl.config import Config
from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from channels import SSHChannelCustom
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_hostname

#woker job
@python_app(executors=["master_htex"])
def runMaster(inputs, hp):
    from modules.model import runRForest

    return runRForest(inputs, hp)

#woker job
@python_app(executors=["tauri_htex"])
def runTauri(inputs, hp):
    from modules.model import runANN

    return runANN(inputs, hp)

    
#woker job
@python_app(executors=["adhafera_htex"])
def runAdhafera(inputs, hp):
    from modules.model import runKNN
    from modules.model import runLinearRegretion

    return runLinearRegretion(inputs, hp)


#master job
def run():
    tauri_hostname = sys.argv[1]
    adhafera_hostname = sys.argv[2]
    port = sys.argv[3]
    username = sys.argv[4]
    passwd = sys.argv[5]
    shared_dir = sys.argv[6]

    parsl.load(executionConfig([tauri_hostname, adhafera_hostname], port, username, passwd, shared_dir))
    print("... config loaded!")

    #data = build_dataset()
    #data_b = pickle.dumps(data)
    print("... dataset loaded!")

    times = []
    for i in range(0, 3):
        ts = datetime.datetime.now()

        #master0 = runMaster(data_b, [*np.arange(0.1, 0.11, 0.1)])

        #master0 = runMaster(None, zip(np.random.randint(low=200, high=1000, size=5), np.random.randint(low=50, high=300, size=5)))
        #master1 = runMaster(data_b, [*range(3, 5)])
        #master2 = runMaster(data_b, [*range(5, 7)])
        #master3 = runMaster(data_b, [*range(7, 9)])
        #master4 = runMaster(data_b, [*range(9, 11)])

        #tauri0 = runTauri(None, zip(np.random.randint(low=80, high=100, size=8), np.random.randint(low=50, high=70, size=8)))
        tauri0 = runTauri(None, zip(np.random.randint(low=80, high=100, size=1), np.random.randint(low=50, high=70, size=1)))
        tauri1 = runTauri(None, zip(np.random.randint(low=80, high=100, size=1), np.random.randint(low=50, high=70, size=1)))
        tauri2 = runTauri(None, zip(np.random.randint(low=80, high=100, size=1), np.random.randint(low=50, high=70, size=1)))
        tauri3 = runTauri(None, zip(np.random.randint(low=80, high=100, size=1), np.random.randint(low=50, high=70, size=1)))
        tauri4 = runTauri(None, zip(np.random.randint(low=80, high=100, size=1), np.random.randint(low=50, high=70, size=1)))
        tauri5 = runTauri(None, zip(np.random.randint(low=80, high=100, size=1), np.random.randint(low=50, high=70, size=1)))
        tauri6 = runTauri(None, zip(np.random.randint(low=80, high=100, size=1), np.random.randint(low=50, high=70, size=1)))
        tauri7 = runTauri(None, zip(np.random.randint(low=80, high=100, size=1), np.random.randint(low=50, high=70, size=1)))

        #adh0 = runAdhafera(data_b, [*np.arange(0.001, 0.01, 0.001)])
        #adh1 = runAdhafera(data_b, [*range(23, 25)])
        #adh2 = runAdhafera(data_b, [*range(25, 27)])
        #adh3 = runAdhafera(data_b, [*range(27, 29)])
        #adh4 = runAdhafera(data_b, [*range(29, 31)])

        print("Waiting results ...")
        outputs = [i.result() for i in [
            #master0, #master1, master2, master3, master4,
            tauri0, tauri1, tauri2, tauri3, tauri4,tauri5, tauri6, tauri7,
            #adh0, #adh1, adh2, adh3, adh4
        ]]

        ts = datetime.datetime.now() - ts
        times.append(ts)
        print(outputs)

    print("times: " + str(times))

    mean = np.mean(times)
    print("elapsed mean: " + str(mean))

    std = np.std([t.total_seconds() for t in times])
    print("elapsed std: " + str(std))


def executionConfig(hostname, port, username, passwd, shared_dir):
    cmd= ("/home/rfialho/.local/bin/process_worker_pool.py {debug} {max_workers} "
                            "-p {prefetch_capacity} "
                            "-c {cores_per_worker} "
                            "-m {mem_per_worker} "
                            "--poll {poll_period} "
                            "--task_url={task_url} "
                            "--result_url={result_url} "
                            "--logdir={logdir} "
                            "--block_id={{block_id}} "
                            "--hb_period={heartbeat_period} "
                            "--hb_threshold={heartbeat_threshold} ")

    timeout = 86400.0

    tauri_sshChannel = SSHChannelCustom(hostname=hostname[0], username=username, password=passwd, port=port, script_dir=shared_dir, timeout=timeout)
    adhafera_sshChannel = SSHChannelCustom(hostname=hostname[1], username=username, password=passwd, port=port, script_dir=shared_dir, timeout=timeout)

    config = Config(
        strategy=None,
        executors=[
            HighThroughputExecutor(
                label="tauri_htex",
                cores_per_worker=1,
                mem_per_worker=2,
                max_workers=10,
                worker_debug=True,
                working_dir= shared_dir,
                worker_logdir_root=shared_dir,
                worker_ports=(7500, 7501),
                interchange_port_range=(48000, 55000),
                address="localhost",
                provider = LocalProvider(
                    channel=tauri_sshChannel,
                    cmd_timeout=timeout,
                    init_blocks=1,
                    nodes_per_block=10,
                    max_blocks=1
                ),
                launch_cmd=(cmd),
            ),
           # HighThroughputExecutor(
            #    label="adhafera_htex",
             #   cores_per_worker=1,
              #  mem_per_worker=0.35,
               # max_workers=6,
                #worker_debug=True,
                #working_dir=shared_dir,
                #worker_logdir_root=shared_dir,
                #worker_ports=(7502, 7503),
                #interchange_port_range=(48000, 55000),
                #address="localhost",
                #provider=LocalProvider(
                #    channel=adhafera_sshChannel,
                #    cmd_timeout=timeout,
                #    max_blocks=1
                #),
                #launch_cmd=(cmd),
            #),
           # HighThroughputExecutor(
           #     label="master_htex",
           #     cores_per_worker=1,
           #     mem_per_worker=0.35,
           #     max_workers=6,
           #     worker_debug=True,
           #     address=address_by_hostname(),
           #     provider=LocalProvider(
           #         channel=LocalChannel(),
           #         cmd_timeout=timeout,
           #         max_blocks=1
           #     ),
           # )
        ])

    return config


if __name__ == '__main__':
    print("> Run ...")
    run()
    print("> Done!")
