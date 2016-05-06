""" Worker conduct the experiment
    Every  class inherit from Worker basic class defines a (set of)
    experiment(s)
    The subclasses of Worker define every python script of experiments 
"""
import abc

class Worker (metaclass=abc.ABCMeta):
    """ Basic class definition for Worker Class """
    def __init__(self):
        """ intial fucntion Worker class """
        print ("initial call of Worker class")

    def _set_simple_numerical_method (self, method_to_call):
        """ numerical_method property setting function"""
        self._simple_numerical_method = method_to_call

    def _get_simple_numerical_method(self):
        """ numerical_method property getting function"""
        return self._simple_numerical_method

    simple_umerical_method = property (_get_simple_numerical_method,
            _set_simple_numerical_method)

    @abc.abstractmethod
    def _setup_testbed(self):
        """ this can considered as a basic experiment input descripting """
        pass

    @abc.abstractmethod
    def _setup_numerical_algorithm(self):
        """ After a linear solver or other numerical methods loaded
            we need to setup the basic prarm for the algorithm
        """
        pass

    @abc.abstractmethod
    def conduct_experiments(self):
        """ function to condution the experiment """
        pass

    @classmethod
    def __subclasshook__(cls,C):
        if cls is Worker:
            attrs = set(dir(C))
            if set(cls.__abstractmethods__)<=attrs:
                return True
        return NotImplemented



def main ():
    """ main function for test Worker class """
    pass
    #worker_test=Worker()
    #worker_test.conduct_experiments()
    #worker_test_derivative = WorkerIterativeLinearSystemSolver()
    #print("Try Inheritance")
    #worker_test_derivative.conduct_experiments()



if __name__ == "__main__":
    """ call main funtion for testing """
    main()
