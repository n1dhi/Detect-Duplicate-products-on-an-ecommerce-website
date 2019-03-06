# Detect-Duplicate-products-on-an-ecommerce-website
Used Autoencoder and KMeans CLustering


For detection of duplicate products on an e commerce website, I decided to go with images of products as their unique identifier.

So, if there are two products with same images, they can easily be considered as duplicates. The data being quite extensive and rich, was not easy to parse. I used **Pandas** library to load the csv file and then **skimage** to read the images of products from their given urls. Inorder to bring uniformity among all the images I resized them all to 3 * 200 * 100 dimension.

Now, the next thing that we could do was to compare images of each product with the images of all the other products, inorder to find out the duplicates. This was quite inefficient and time taking process. So, I decided to apply **KMeans** clustering on the images  and later compare the products in same clusters. But before grouping them into clusters, I wanted to encode them into smaller size vectors so that applying **KMeans** could be easier.

What better way to encode an image than an **Autoencoder** !! So I applied a **Convolutional AutoEncoder** on all my images and converted them to 1*256 size vectors.

The architecture of the autoencoder was:
**Encoder** :
	Conv2d->Relu->Conv2d->MaxPool->Linear->Linear
	
**Decoder** :
	Linear->Linear->MaxUnpool->Conv2d->Conv2d

I trained this autoencoder on 50,000 images of **Tops** and it took around 9 hours to train on a **K80 Tesla GPU**

I used Mean Squared error as loss and trained for 3 epochs. Loss on training data was 0.0273

After, encoding all the images of **Tops** , I applied KMeans Clustering on them. Then after passing each image encoding through the Kmeans model I could label them into different clusters (added a new column of cluster in the dataset).

Then I compared all the products in one cluster to all the other products in that cluster (considerably reducing the comparisons from N*N to               N * size_cluster.

I used 200 clusters here and trained the Kmeans clustering model on 10000 products.

After encoding and clustering I compared different products in a cluster, and products with L2 norm (L2 difference between their encodings) as 0 were considered as duplicates.

Due to timing constraints I could only perform the comparisons among 10,000 products. 

Now if we get a new product and we wish to find if duplicate product exists, we only need to find its encoding and cluster and then compare it with products in its own cluster.

Here one of the challenges we might face is if two models are wearing same tops, as in the products are same but the models wearing it might be different. In this case the above approach will not result in their L2 norm being 0 and hence they will not be identified as duplicates. For now we did not have any such data so we are good to go. But if we face such a scenario in future I thought of applying semantic segmentation on the product images so as to get the image of only the product and then we can apply the model described above ( AutoEncoder -> KMeans).

Even now, we can handle such a scenario by keeping the threshold for L2 norm slightly greater than 0 but that will hamper our accuracy.

I am handing in the results I obtained for 10,000 images. I'll train this model further with increased dataset and I'll get the duplicates for all 346927 tops (It will take atleast another 9 hours).  I would've implemented the semantic segmentation as well but with the timing constraints and the dataset we have currently, it seemed like an overkill. But I'll surely implement it in the future so that our model is robust to scenarios mentioned above.



