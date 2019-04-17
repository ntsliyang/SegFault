import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import torch


# Plot diagrams
# Create matplotlib figure and subplot axes
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 12), constrained_layout=True)
fig.suptitle("Training stats")

def plot_durations(episode_rewards=None, ex_valnet_loss=None, in_valnet_loss=None, hash_loss=None, idx_range=None, mean_interval=5):
    """Plot diagrams for episode durations and value net loss"""
    global fig, ax1, ax2, ax3

    if idx_range is not None:
        start_idx, end_idx = idx_range
        x_axis = range(start_idx, end_idx)


    if episode_rewards is not None:
        # Plot episode duration
        durations_t = torch.tensor(episode_rewards, dtype=torch.float)
        ax1.cla()
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Rewards')
        ax1.grid()
        if idx_range is None:
            ax1.plot(durations_t.numpy())
        else:
            ax1.plot(x_axis, durations_t.numpy()[start_idx : end_idx])

        # Take 100 episode averages and plot them too
        if len(durations_t) > mean_interval:
            means = durations_t.unfold(0, mean_interval, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(mean_interval - 1), means))
            if idx_range is None:
                ax1.plot(means.numpy())
            else:
                ax1.plot(x_axis, means.numpy()[start_idx : end_idx])


    if ex_valnet_loss is not None:
        # Plot episode value net loss
        ax2.cla()
        ax2.set_title('Extrinsic ValueNet Loss')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss value')
        ax2.grid()
        if idx_range is None:
            ax2.plot(ex_valnet_loss)
        else:
            ax2.plot(x_axis, ex_valnet_loss[start_idx : end_idx])

    if in_valnet_loss is not None:
        # Plot episode value net loss
        ax3.cla()
        ax3.set_title('Intrinsic ValueNet Loss')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss value')
        ax3.grid()
        if idx_range is None:
            ax3.plot(in_valnet_loss)
        else:
            ax3.plot(x_axis, in_valnet_loss[start_idx : end_idx])

    if hash_loss is not None:
        # Plot episode value net loss
        ax4.cla()
        ax4.set_title('Hash Loss')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Loss value')
        ax4.grid()
        if idx_range is None:
            ax4.plot(hash_loss)
        else:
            ax4.plot(x_axis, hash_loss[start_idx: end_idx])

    # Re-draw, show, and give the system a bit of time to display and refresh the window
    plt.draw()
    plt.show()
    plt.pause(0.01)

