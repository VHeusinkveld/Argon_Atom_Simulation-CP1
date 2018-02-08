import numpy as np
from scipy import integrate

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation

def make_3d_animation(L, pos, delay=10, initial_view=(30, 20),
                      rotate_on_play=0.5):
    """Create a matplotlib animation object to visualize
    the motion of particles in 3D.

    Parameters:
    -----------
    L: float
        size of the simulation box
    pos: array of size (n_tstep, N, 3)
        The time-dependence of the N particles. The first index of the
        array corresponds to the time.
    delay: float
        Delay between frames of the final movie in milliseconds.
        Default is 10.
    initial_view: tuple of two floats
        Initial view of the 3D box of the form (altitude degrees,
        az imuth degrees). Defaults to (30, 20).
    rotate_on_play: float
        angle (in degrees) with which the view is rotated (the azimuth
        angle is increased) by every frame. Default is 0.5.

    Results:
    --------
    anim: animation object
        Returns a matplotlib animation object. In an IPython notebook, the
        animation can be displayed as:

            from IPython.display import HTML

            anim = make_3d_animation(...)
            HTML(anim.to_html5_video())

        From a regular python script, you can call

            import matplotlib.pyplot as plt

            anim = make_3d_animation(...)
            plt.show()
    """
    # Set up figure & 3D axis for animation
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d', aspect="equal")
    ax.axis('off')

    pts = ax.plot(xs=[], ys=[], zs=[], marker="o", linestyle='None')
    
    # plot a bounding box
    ax.plot(xs=[0, L, L, 0, 0],
            ys=[0, 0, L, L, 0],
            zs=[0, 0, 0, 0, 0], color="k")
    ax.plot(xs=[0, L, L, 0, 0],
            ys=[0, 0, L, L, 0],
            zs=[L, L, L, L, L], color="k")
    ax.plot(xs=[0, 0], ys=[0, 0], zs=[0, L], color="k")
    ax.plot(xs=[0, 0], ys=[L, L], zs=[0, L], color="k")
    ax.plot(xs=[L, L], ys=[0, 0], zs=[0, L], color="k")
    ax.plot(xs=[L, L], ys=[L, L], zs=[0, L], color="k")

    # prepare the axes limits
    ax.set_xlim((0, L))
    ax.set_ylim((0, L))
    ax.set_zlim((0, L))

    # set point-of-view: specified by (altitude degrees, azimuth degrees)
    ax.view_init(*initial_view)

    # initialization function: plot the background of each frame
    def init():
        return pts

    # animation function.  This will be called sequentially with the frame number
    def animate(i):
        pts[0].set_data(pos[i, :, 0], pos[i, :, 1])
        pts[0].set_3d_properties(pos[i, :, 2])

        ax.view_init(initial_view[0], initial_view[1] + rotate_on_play * i)
        fig.canvas.draw()
        return pts

    # instantiate the animator.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(pos), interval=delay, blit=True)

    return anim
