# Product Feature Clustering
Utilising K-Means clustering to cluster products based on features, price, and reviews to recommend alternatives or group by similarity, using the Amazon Product dataset.
I will attempt to develop the algorithms from scratch in order to solidify my learning and then use libraries such as scikitlearn to cross-reference my results

* _Whatever I learn and any notes I make are in the notes.txt file_

🔍 Project Description
This project aims to apply K-Means clustering to group similar products on an e-commerce platform based on their features. In large-scale online stores, customers often face difficulty navigating through a vast number of products. By clustering products based on key attributes (such as price, ratings, reviews, and category), we can:
- Improve product recommendation systems
- Assist users in finding similar items
- Help sellers identify market competition
- Optimize inventory and pricing strategies
- The clustering model will uncover hidden patterns in product data that are not immediately obvious and can enhance decision-making in marketing and operations.

🎯 Project Aims
- Preprocess and structure e-commerce product data for clustering, ensuring high quality and consistency.
- Select relevant features (e.g., price, rating, number of reviews, discount, brand, product category) that influence customer decisions.
- Apply K-Means clustering to group similar products together.
- Visualize and interpret clusters to identify distinct product segments.
- Evaluate the clustering results using metrics like the elbow method or silhouette score.
- Explore practical applications such as:
- Recommending similar products
- Detecting premium vs. budget items
- Category-specific inventory planning

🧰 Potential Features for Clustering
- Price
- Rating (e.g., 1-5 stars)
- Number of reviews
- Discount %
- Brand popularity
- Category/type

📦 Example Use Cases
- A user views a laptop → recommend other laptops in the same cluster with similar price/specs.
- A seller wants to enter a market segment → analyze which clusters are highly competitive.
- The system identifies a group of products with poor ratings but high prices → flag for review.
