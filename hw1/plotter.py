import seaborn as sns
import matplotlib.pyplot as plt

class Plotter:

    @staticmethod
    def plot(name, epochs, losses, rewards, steps):
        if X_SERVER:
            fig = plt.figure()
            plt.title("Loss, reward and steps taken by epoch")
            plt.axis("off")

            #df = pd.DataFrame()
            #df['losses'] = losses
            #df['rewards'] = rewards
            fig.add_subplot(1,2,1)
            sns.tsplot(time=range(len(losses)), data=losses, linestyle='-')
            fig.add_subplot(1,2,2)
            sns.tsplot(time=range(len(rewards)), data=np.reshape(rewards, -1), linestyle="-")
            fig.add_subplot(1,3,1)
            sns.tsplot(time=range(len(steps)), data=steps, linestyle="-")

            plt.savefig("output/{0}_{1}".format(name, epochs))

