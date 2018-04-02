import numpy as np
import csv


#some lists that are useful later
#letter_distribution=np.array([.08167,.01492,.02782,.04253,.12702,.02228,.02015,.06094,.06966,.00153,.00772,.04025,.02406,.06749,.07507,.01929,.00095,.05987,.06327,.09056,.02758,.00978,.02361,.00150,.01974,.00074])
#letters=np.array(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])

#Temporarily removed the R and the G, and boosted @ by 0.002 (up to 0.1081) to compensate. Then boosted from 0.1081 to 0.1099 to fill the missing space which exists for some reason now?
letter_distribution=np.array([0.1099, 0.0711, 0.0694, 0.0691, 0.0632, 0.0475, 0.0421, 0.0396, 0.0361, 0.0318, 0.0295, 0.0286, 0.0276, 0.0276, 0.0215, 0.0210, 0.0201, 0.0195, 0.0193, 0.0180, 0.0179, 0.0174, 0.0171, 0.0150, 0.0145, 0.0140, 0.0125, 0.0118, 0.0099, 0.0097, 0.0081, 0.0080, 0.0059, 0.0056, 0.0050, 0.0043, 0.0041, 0.0010, 0.0007, 0.001, 0.001, 0.001, 0.001, 0.001])
letters = np.array(['@', 'n', 'r', 't', 'I', 's', 'd', 'l', 'i', 'k', 'D', 'e', 'm', 'z', 'p', '&', 'v', 'w', 'u', 'b', '3', 'V', 'f', 'y', '0', 'h', 'O', 'A', 'N', 'S', 'j', 'g', 'J', 'C', 'W', 'U', 'T', 'Y', 'Z', 'H', 'o', 'E', 'a', '2'])

#generate letter to number dictionary
def generate_dictionary(location):
    with open(location,'rb') as my_file:
        reader=csv.reader(my_file)
        char_to_nums={}
        char_to_nums['\t']=0 # we use this as a special character later
        for row in reader:
            for letter in row[0]:
                if letter not in char_to_nums.keys():
                    char_to_nums[letter]=len(char_to_nums)
        my_file.close()
    return char_to_nums
#generate the reverse dictionary
def reverse_dictionary(dictionary):
    rev_dict={v: k for k, v in dictionary.iteritems()}
    return rev_dict

#convert poems to matrices using 1-k encoding
def poem_to_mat(poem,dictionary):
    poem_length=len(poem)
    vocab_length=len(dictionary)
    poem_mat=np.zeros((poem_length,vocab_length))
    vocab_list=[dictionary[s] for s in poem]
    poem_mat[xrange(poem_length),vocab_list]=1
    return poem_mat

#generate labels for poems
def generate_labels(poem_mat,dictionary):
    labels=np.zeros(poem_mat.shape[0])
    labels[:-1]=np.argmax(poem_mat,axis=1)[1:]
    labels[-1]=dictionary['\t'] #label last character as tab to indicate the end has been reached.
    return labels

#convert a batch of poems to a 3-tensor along with a tensor of labels and a mask
def poem_batch_to_tensor(X,y=None):
    b=len(X)
    v=len(X[0][0])
    #print b,v
    len_list=np.array([len(X[i]) for i in range(b)])
    m=np.max(len_list)
    num_chars=np.sum(len_list) # useful for
    #pad with zeros so they are all same length
    X_mat=np.array([np.vstack((np.array(X[i]),np.zeros((m-len_list[i],v)))) for i in range(b)])
    X_mat=np.swapaxes(X_mat,0,1)
    y_mat=np.array([np.hstack((np.array(y[i]),np.zeros(m-len_list[i]))) for i in range(b)],dtype=int).T
    #create a mask of so that later we only accumulate cost for entries that actually correspond to letters
    mask=np.ones((m,b))
    for i in range(b):
        mask[len_list[i]:,i]=0
    return X_mat, mask, num_chars,y_mat
