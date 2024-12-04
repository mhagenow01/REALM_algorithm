from abc import ABC, abstractmethod

# abstract base class
class AssistanceMode(ABC):
    @abstractmethod
    def getPenalizedLikelihood(self, actions):
        # actions are num_rollouts x horizon x num_states
        pass

    @abstractmethod
    def getNumberOfModeParameters(self,actions):
        pass
