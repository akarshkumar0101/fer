import jax.numpy as jnp

def hsv2rgb(h, s, v):
    """
    Convert HSV to RGB.
    All inputs and outputs should be in the range [0, 1].
    """
    h = h * 360.

    c = v * s
    x = c * (1 - jnp.abs((h / 60) % 2 - 1))
    m = v - c

    r1, g1, b1, c1 = c, x, 0, (0 <= h)*(h<60)
    r2, g2, b2, c2 = x, c, 0, (60 <= h)*(h<120)
    r3, g3, b3, c3 = 0, c, x, (120 <= h)*(h<180)
    r4, g4, b4, c4 = 0, x, c, (180 <= h)*(h<240)
    r5, g5, b5, c5 = x, 0, c, (240 <= h)*(h<300)
    r6, g6, b6, c6 = c, 0, x, (300 <= h)*(h<360)

    r = r1 * c1 + r2 * c2 + r3 * c3 + r4 * c4 + r5 * c5 + r6 * c6
    g = g1 * c1 + g2 * c2 + g3 * c3 + g4 * c4 + g5 * c5 + g6 * c6
    b = b1 * c1 + b2 * c2 + b3 * c3 + b4 * c4 + b5 * c5 + b6 * c6
    r, g, b = r + m, g + m, b + m
    return r.clip(0., 1.), g.clip(0., 1.), b.clip(0., 1.)