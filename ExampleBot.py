from tools import  *
from objects import *
from routines import *
import os
import logging

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir, 'ocrlbot.log')
logging.basicConfig(filename=log_file_path, level=logging.DEBUG)


#This file is for strategy

class ExampleBot(GoslingAgent):
    def run(agent, packet):
        # logging.debug(f"Car Location: {agent.me.location}")
        logging.debug(f"Enemy Pose: {agent.foes[0].location}; {packet.game_cars[agent.foes[0].index].physics.rotation}; "
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

