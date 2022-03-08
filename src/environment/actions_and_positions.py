from enum import Enum


class Actions(Enum):
    Buy = 0
    Hold = 1
    Sell = 2


class Positions(Enum):
    Long = 1
    Neutral = 0
    Short = -1
