import matplotlib.pyplot as plt
import numpy as np 

plt.rcParams.update({'font.size': 24})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

prefix = "big"
prefix2 = "huge"

scores_muffliato = np.load("result/"+prefix+"muffliato.npy")
scores_central = np.load("result/"+prefix+"central.npy")
privacy = np.load("result/"+prefix+"privacy.npy")
privacy2 = np.load("result/"+prefix2+"privacy.npy")


n_iter = scores_central.shape[1]
beta=1

iter_list = [i for i in range(n_iter)]

fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]},figsize=(7,10))
fig.subplots_adjust( hspace=.35)

curve = 1
histo = 0

right_side = axs[curve].spines["right"]
right_side.set_visible(False)
up_side = axs[curve].spines["top"]
up_side.set_visible(False)

axs[curve].set_ylim([0.5,1])

axs[curve].errorbar(iter_list, scores_muffliato.mean(axis=0),  beta* scores_muffliato.std(axis=0), label=r"Muffliato", color="g", capthick=3, capsize = 4, lw=3)

axs[curve].errorbar(iter_list, scores_central.mean(axis=0),  beta* scores_central.std(axis=0), label=r"Central", color="b", capthick=3, capsize = 4, lw=3)
axs[curve].set_xlabel('Iterations')
axs[curve].set_ylabel('Accuracy')
axs[curve].legend(loc='lower right')


axs[histo].hist(privacy2.flatten()/privacy[0, 0], bins=80, color='g', label="PNDP (n=4000)")
axs[histo].hist(privacy.flatten()/privacy[0, 0], bins=80, color='xkcd:dark green', label="PNDP (n=2000)")

axs[histo].set_yticks([0,10000])
axs[histo].set_yticklabels([0,r"$10^3$"])

axs[histo].axvline(1, label = "LDP", c='r')

axs[histo].set_xlabel("Privacy loss")
axs[histo].set_ylabel('Nodes')
axs[histo].legend(loc='upper center')

fig.savefig(prefix+"summary.pdf", bbox_inches='tight', pad_inches=0)
plt.show()
