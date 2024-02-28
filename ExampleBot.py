from tools import  *
from objects import *
from routines import *
from datetime import datetime
import os
import logging

timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"ocrlbot-{timestamp}.log")

logger = logging.getLogger("ocrl_logger")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(file_handler)


#This file is for strategy

class ExampleBot(GoslingAgent):
    def run(agent, packet):
        # logging.debug(f"Car Location: {agent.me.location}")
        logger.debug(f"Enemy Pose: {agent.foes[0].location}; {packet.game_cars[agent.foes[0].index].physics.rotation}; "
                     f"{agent.foes[0].velocity}; {agent.foes[0].angular_velocity}")

        # comment out to stop the bot from driving off and triggering a respawn
        #An example of using raw utilities:
        # relative_target = agent.ball.location - agent.me.location
        # local_target = agent.me.local(relative_target)
        # defaultPD(agent, local_target)
        # defaultThrottle(agent, 2300)
        #
        # #An example of pushing routines to the stack:
        # if len(agent.stack) < 1:
        #     if agent.kickoff_flag:
        #         agent.push(kickoff())
        #     else:
        #         agent.push(atba())

