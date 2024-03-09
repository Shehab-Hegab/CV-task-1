def histogram(img):
    if len(img.shape) == 3:
        return rgb_histogram(img)
    else:
        return gray_histogram(img)
    
def channel_histogram(img):
    rows,cols = img.shape
    hist= np.zeros(256)
    for row in range(rows):
        for col in range(cols):
            intensity = int(img[row][col])
            hist[intensity]+=1
    return hist

def gray_histogram(img):
    hist = channel_histogram(img)
    fig = plt.figure(figsize =(15, 7))
    plt.bar(range(256),hist,color='gray')
    plt.xticks(np.arange(0, 256, 10))
    plt.show;
    return hist

def rgb_histogram(img):
    r_hist = channel_histogram(img[:,:,0])
    g_hist = channel_histogram(img[:,:,1])
    b_hist = channel_histogram(img[:,:,2])
    hist_dict= {}
    hist_dict['R']= r_hist
    hist_dict['G']= g_hist
    hist_dict['B']= b_hist
    
    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].bar(range(256),r_hist,color='red')
    ax[1].bar(range(256),g_hist,color='green')
    ax[2].bar(range(256),b_hist,color='blue')
    fig.suptitle("RGB Distribution")
    plt.show;
    
    return hist_dict

