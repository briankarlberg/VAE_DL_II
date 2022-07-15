plt.plot(list(range(1,7)), history_dict['kl_loss'])
plt.title("KL loss, coding gene encoding")
plt.xlabel('Epoch')
plt.ylabel('Kullback-Leibler divergence')

plt.annotate('2022-07-14',(4,50))
plt.annotate('Run time 2 minutes', (4,45))
plt.annotate('Memory call: xxGB', (4,40))
plt.annotate('Number of GPUs: 1', (4,35))
plt.annotate('File size: 16.6 MB', (4,30))

plt.savefig('coding_genes_KL.png')