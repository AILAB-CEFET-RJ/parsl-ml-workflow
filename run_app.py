import parsl
from parsl.app.app import python_app, bash_app
from parsl.configs.local_threads import config


@python_app
def runANN():
    return "Run an ANN!"


def run():
    parsl.load(config)

    print(runANN().result())


if __name__ == '__main__':
    print("> Run ...")
    run()
    print("> Done!")
