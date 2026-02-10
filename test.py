from clashroyalebuildabot.actions import (
    BabyDragonAction,
    FireballAction,
    GiantAction,
    KnightAction,
    ZapAction,
)
from clashroyalebuildabot.actions.archers_action import ArchersAction
from clashroyalebuildabot.actions.musketeer_action import MusketeerAction
from clashroyalebuildabot.actions.skeletons_action import SkeletonsActions
from clashroyalebuildabot.bot import Bot
from clashroyalebuildabot.gui.utils import load_config

actions = [
    ZapAction,
    FireballAction,
    BabyDragonAction,
    SkeletonsActions,
    KnightAction,
    ArchersAction,
    MusketeerAction,
    GiantAction,
]

config = load_config()
bot = Bot(actions=actions, config=config)

bot.run()
