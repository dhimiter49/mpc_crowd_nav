import numpy as np
import matplotlib.pyplot as plt


def plot_in_time(ctrl, env, dt):
    ctrl[0].agent_pos.append(env.get_wrapper_attr("current_pos").copy())
    ctrl[0].crowd_pos.append(env.get_wrapper_attr("crowd_pos_vel")[0].copy()[0])
    agent_x = np.array(ctrl[0].agent_pos).copy().reshape(-1, 2)[:, 0].flatten()
    crowd_x = np.array(ctrl[0].crowd_pos).copy().reshape(-1, 2)[:, 0].flatten()

    # Sample data
    t_y = np.arange(len(agent_x)) * dt

    # Create the plot
    plt.plot(agent_x, t_y, label='Agent', color='k')
    plt.fill_betweenx(t_y, agent_x - 0.4, agent_x + 0.4, color='k', alpha=0.2)
    plt.plot(crowd_x, t_y, label='Crowd', color='r')
    plt.fill_betweenx(t_y, crowd_x - 0.4, crowd_x + 0.4, color='r', alpha=0.2)

    # Add labels and legend
    plt.xlabel('Space [m]')
    plt.ylabel('Time [s]')
    plt.title('Crowd at 1.4m/s')
    plt.legend()

    plt.show()
