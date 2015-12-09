import network
import numpy as np

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def vectorized_data(arr):
	e = np.zeros((1024, 1))
	count = 0
	for item in arr:
		e[count][0] = item
		count = count + 1
	return e/255

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

a = unpickle("data_batch_1") 
t = unpickle("test_batch") 



dataR = []
for item in a.get('data'):
	dataR.append(vectorized_data(item[0:1024]))

label = []
for item in a.get('labels'):
	label.append(vectorized_result(item))


dataRt = []
for item in t.get('data'):
	dataRt.append(vectorized_data(item[0:1024]))	

# labelt = []
# for item in t.get('labels'):
# 	label.append(vectorized_result(item))




training_data = [(x, y) for x,y in zip(dataR,  a.get("labels"))]

test_data = [(x, y) for x,y in zip(dataRt,  t.get("labels"))]
f = open("subtest.txt","w")

subtrain_data = training_data[0:500]
subtest_data = test_data[0:500]

for item in subtest_data:
	f.write(str(item[1]))
	for i in range(len(item[0])):
	   f.write(" "+str(i+1)+":"+str(item[0][i][0]))
	# for i in range(len(item[0])):
	# 	f.write(str(i+1)+":"+str(item[0][i])+" ")

	f.write("\n")

f.close()

