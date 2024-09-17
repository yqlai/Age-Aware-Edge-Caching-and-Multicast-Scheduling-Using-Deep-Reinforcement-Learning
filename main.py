from Queues import User_Request_Queue
from SBS import SBS, Decision_Making
from MBS import MBS

if __name__ == '__main__':
    num_content = 30
    cache_size = 5
    
    sbs = SBS(cache_size, num_content)
    mbs = MBS()

    sbs.initialize()

    reward = 0

    num_time_slots = 10000
    time_slot = 0
    
    Decision_Making(mbs, sbs, num_time_slots)

    print('End of Simulation')