
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
    # e = np.zeros((1024, 1))
    # count = 0
    # for item in arr:
    #   e[count][0] = item
    #   count = count + 1
    # return e/255

    e = np.zeros((1024, 1))
    r = np.zeros((1024, 1))
    g = np.zeros((1024, 1))
    b = np.zeros((1024, 1))  

    count = 0
    for item in arr:
        if count <= 1023:
            r[count][0] = item
        elif count <= 2047 and count > 1023:
            g[count-1024][0] = item
        else:
            b[count-1024*2][0] = item
        count = count + 1

    e = 0.299*r + 0.587*g + 0.114*b
    
    return e/255.0

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
    dataR.append(vectorized_data(item[0:]))

label = []
for item in a.get('labels'):
    label.append(vectorized_result(item))


dataRt = []
for item in t.get('data'):
    dataRt.append(vectorized_data(item[0:]))    

# labelt = []
# for item in t.get('labels'):
#   label.append(vectorized_result(item))




training_data = [(x, y) for x,y in zip(dataR,  a.get("labels"))]

test_data = [(x, y) for x,y in zip(dataRt,  t.get("labels"))]
f = open("train_grey.txt","w")

# subtrain_data = training_data[0:500]
# subtest_data = test_data[0:500]

for item in training_data:
    f.write(str(item[1]))
    for i in range(len(item[0])):
       f.write(" "+str(i+1)+":"+str(item[0][i][0]))
    # for i in range(len(item[0])):
    #   f.write(str(i+1)+":"+str(item[0][i])+" ")

    f.write("\n")

f.close()

