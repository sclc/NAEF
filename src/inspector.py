""" class Inspector def """
from worker import Worker

class Inspector():
    """ Inspector def """
    def __init__(self, worker):
        """ init Inspector """
        self._inspecting_target = worker

def main ():
    """ main funciton for testing """
    worker_a = Worker()
    inspector_a = Inspector(worker_a)

if __name__ == "__main__":
    main()
