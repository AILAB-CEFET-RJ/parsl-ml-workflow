import sys
import parsl
import pickle
import datetime
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from parsl.app.app import python_app
from parsl.config import Config
from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from channels import SSHChannelCustom
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_hostname

#woker job
@python_app(executors=["master_htex"])
def runMaster(inputs={}, ks=[]):
    from modules.model import runKNN

    return runKNN(inputs, ks)

#woker job
@python_app(executors=["tauri_htex"])
def runTauri(inputs={}, ks=[]):
    from modules.model import runKNN

    return runKNN(inputs, ks)

#woker job
@python_app(executors=["adhafera_htex"])
def runAdhafera(inputs, ks):
    from modules.model import runKNN

    return runKNN(inputs, ks)

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

    data = build_dataset()
    data_b = pickle.dumps(data)
    print("... dataset loaded!")

    times = []
    for i in range(0, 3):
        ts = datetime.datetime.now()

        master0 = runMaster(data_b, [*range(1, 3)])
        master1 = runMaster(data_b, [*range(3, 5)])
        master2 = runMaster(data_b, [*range(5, 7)])
        master3 = runMaster(data_b, [*range(7, 9)])
        master4 = runMaster(data_b, [*range(9, 11)])

        tauri0 = runTauri(data_b, [*range(11, 13)])
        tauri1 = runTauri(data_b, [*range(13, 15)])
        tauri2 = runTauri(data_b, [*range(15, 17)])
        tauri3 = runTauri(data_b, [*range(17, 19)])
        tauri4 = runTauri(data_b, [*range(19, 21)])

        adh0 = runAdhafera(data_b, [*range(16, 23)])
        adh1 = runAdhafera(data_b, [*range(23, 25)])
        adh2 = runAdhafera(data_b, [*range(25, 27)])
        adh3 = runAdhafera(data_b, [*range(27, 29)])
        adh4 = runAdhafera(data_b, [*range(29, 31)])

        print("Waiting results ...")
        outputs = [i.result() for i in [
            master0, master1, master2, master3, master4,
            tauri0, tauri1, tauri2, tauri3, tauri4,
            adh0, adh1, adh2, adh3, adh4
        ]]

        print(outputs)
        ts = datetime.datetime.now() - ts
        times.append(ts)

    print("times: " + str(times))

    mean = np.mean(times)
    print("elapsed mean: " + str(mean))

    std = np.std([t.total_seconds() for t in times])
    print("elapsed std: " + str(std))



def build_dataset():
    train_file = 'datasets/redshifts.csv'

    x = np.loadtxt(train_file, usecols=(range(1, 6)), unpack=True, delimiter=',', dtype='float32').T
    y = np.loadtxt(train_file, unpack=True, usecols=(11), delimiter=',', dtype='float32')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    x_val = x_val.astype(np.float32)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    x_val = scaler.fit_transform(x_val)

    return {
        "x_train": x_train, "y_train": y_train,
        "x_test": x_test, "y_test": y_test,
        "x_val": x_val, "y_val": y_val
    }

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

    tauri_sshChannel = SSHChannelCustom(hostname=hostname[0], username=username, password=passwd, port=port, script_dir=shared_dir)
    adhafera_sshChannel = SSHChannelCustom(hostname=hostname[1], username=username, password=passwd, port=port, script_dir=shared_dir)

    config = Config(
        executors=[
            HighThroughputExecutor(
                label="tauri_htex",
                cores_per_worker=1,
                mem_per_worker=0.35,
                max_workers=6,
                worker_debug=True,
                working_dir= shared_dir,
                worker_logdir_root=shared_dir,
                worker_ports=(7500, 7501),
                interchange_port_range=(48000, 55000),
                address="localhost",
                provider = LocalProvider(
                     channel=tauri_sshChannel
                ),
                launch_cmd=(cmd),
            ),
            HighThroughputExecutor(
                label="adhafera_htex",
                cores_per_worker=1,
                mem_per_worker=0.35,
                max_workers=6,
                worker_debug=True,
                working_dir=shared_dir,
                worker_logdir_root=shared_dir,
                worker_ports=(7502, 7503),
                interchange_port_range=(48000, 55000),
                address="localhost",
                provider=LocalProvider(
                    channel=adhafera_sshChannel
                ),
                launch_cmd=(cmd),
            ),
            HighThroughputExecutor(
                label="master_htex",
                cores_per_worker=1,
                mem_per_worker=0.35,
                max_workers=6,
                worker_debug=True,
                address=address_by_hostname(),
                provider=LocalProvider(
                    channel=LocalChannel()
                ),
            )
        ])

    return config


if __name__ == '__main__':
    print("> Run ...")
    run()
    print("> Done!")
