import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np

prefixs = ["l", "la", "lb", "lc", "ld", "le", "lf", "lg"]

def name_from_prefix(prefix):
    error = [np.load("result/"+prefix+"errornodropout.npy"),
    np.load("result/"+prefix+"errorfewdropout.npy"),
    np.load("result/"+prefix+"errormiddropout.npy"),
    np.load("result/"+prefix+"errorhighdropout.npy")
    ]
    privacy = [np.load("result/"+prefix+"privacynodropout.npy"),
    np.load("result/"+prefix+"privacyfewdropout.npy"),
    np.load("result/"+prefix+"privacymiddropout.npy"),
    np.load("result/"+prefix+"privacyhighdropout.npy")
    ]
    return error, privacy

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 24})

plt.figure()



for prefix in prefixs:
    error, privacy = name_from_prefix(prefix)

    if prefix == "l":
        plt.plot(np.arange(error[0].shape[0]), error[0], label='No dropout', color='xkcd:black')
        plt.plot(np.arange(error[1].shape[0]), error[1], label='10% dropout',  color='xkcd:deep green')
        plt.plot(np.arange(error[2].shape[0]), error[2], label='50% dropout',  color='xkcd:jade')
        plt.plot(np.arange(error[3].shape[0]), error[3], label='90% dropout', color='xkcd:light aqua')
    else:
        plt.plot(np.arange(error[0].shape[0]), error[0], color='xkcd:black')
        plt.plot(np.arange(error[1].shape[0]), error[1],  color='xkcd:deep green')
        plt.plot(np.arange(error[2].shape[0]), error[2],  color='xkcd:jade')
        plt.plot(np.arange(error[3].shape[0]), error[3], color='xkcd:light aqua')

plt.xlabel("Time step")
plt.ylabel("Error")
plt.xscale('log')
plt.yscale('log')
plt.ylim((1, 2000))

plt.legend(bbox_to_anchor = (1.05, 0.6))
plt.savefig("error_dropout.pdf", bbox_inches='tight', pad_inches=0)


plt.figure()

sum_privacy = [[],[],[],[]]

for prefix in prefixs:
    error, privacy = name_from_prefix(prefix)

    if prefix == "l":
        plt.plot(np.arange(privacy[0].shape[0]),privacy[0], label='No dropout' ,color='xkcd:black')

        plt.plot(np.arange(privacy[1].shape[0]),privacy[1], label='Low dropout',color='xkcd:deep green')
        plt.plot(np.arange(privacy[2].shape[0]),privacy[2], label='Medium dropout',color='xkcd:jade')
        plt.plot(np.arange(privacy[3].shape[0]),privacy[3], label='High dropout',color='xkcd:light aqua')


    else: 
        plt.plot(np.arange(privacy[0].shape[0]),privacy[0] ,color='xkcd:black')

        plt.plot(np.arange(privacy[1].shape[0]),privacy[1],color='xkcd:deep green')
        plt.plot(np.arange(privacy[2].shape[0]),privacy[2],color='xkcd:jade')
        plt.plot(np.arange(privacy[3].shape[0]),privacy[3],color='xkcd:light aqua')
    
    for i in range(4):
        sum_privacy[i].append(privacy[i][-1])


for i in range(4):
    print(np.mean(np.array(sum_privacy[i])), np.std(np.array(sum_privacy[i])))
    print()

plt.xlabel("Time step")
plt.ylabel("Privacy loss")
plt.xscale('log')
#plt.legend(bbox_to_anchor = (1.05, 0.6))
plt.savefig("privacyloss_dropout.pdf", bbox_inches='tight', pad_inches=0)

plt.show()
