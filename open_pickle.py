import pickle


filename1 = 'artworks_urls.pkl'
f1 = open(filename1, 'rb')
data=pickle.load(f1)
f1.close()
print('number of urls', len(data))
for i in range(10):
    print(data[i])
# save skipped urls
filename2 = 'skipped_urls.pkl'
f2 = open(filename2, 'rb')

data=pickle.load(f2)
f2.close()
print('number of urls', len(data))
#for i in range(4):
print(data)
    