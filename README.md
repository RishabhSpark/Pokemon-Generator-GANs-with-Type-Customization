# Pokemon-Generator-GANs-with-Types
## Abstract
This project report presents a novel approach to generating video game (Pokemon) and anime character concepts using Generative Adversarial Networks (GANs). The primary objective of this research is to leverage the power of GANs to generate diverse and visually appealing anime characters and Pokemon, while also providing users with the ability to customize their creations based on specific Pokemon types. The project consists of two main components: anime character generation and Pokemon generation. For the first phase of Anime Character Generation, the project involves training a GAN on a dataset of anime characters to generate new and unique character designs. The GAN consists of a generator network that generates images and a discriminator network that distinguishes between real and generated images. Through an adversarial training process, the generator network learns to produce increasingly realistic and visually appealing anime character concepts. In the second phase, another GAN model is trained specifically for generating Pokémon designs. This model takes inspiration from existing Pokémon designs and generates novel Pokémon that exhibit similar characteristics, like types. The generated Pokémon designs are then passed through a deep learning model that labels them with their most relevant types. This labeling model utilizes a large dataset consisting of all pre-existing Pokémon with their respective types to accurately predict the top types for each generated Pokémon. By incorporating this process, users can specify desired Pokémon types and receive generated Pokémon designs that match their preferences. This enhances the user's ability to create customized Pokémon based on specific types, allowing for more immersive and personalized concept art generation for artists. The results of this project demonstrate the effectiveness of using GANs to generate new and unique designs; and showcase a wide range of visually diverse and appealing concepts. The incorporation of type labeling further enhances the usability and personalization of the system, enabling users to create unique Pokémon based on their preferred types.

**_Keywords —_** GAN, Deep Learning, Image Processing, Character Generation, Pokémon Generation

## Introduction 
Character design is vital in the realms of video games and anime for engrossing viewers and transporting them to realms of fantasy. Be it anime characters or Pokémon, developing aesthetically appealing and unique characters demands an immense amount of imagination and skill. However, given that there are already countless characters, it becomes quite difficult to come up with a character or Pokémon design. This makes the process of coming up with new thoughts typically very tough and time-consuming. Hence, the idea of using deep learning techniques for generating new designs to help artists get new inspiration can be beneficial and help reduce the time required.

Recent developments in artificial intelligence and machine learning have created new opportunities for creating unique and diverse character conceptions. Among multiple different techniques, Generative Adversarial Networks (GANs) have attracted a lot of attention. GANs have shown an impressive ability in producing realistic and high-quality pictures, in a variety of fields, including computer vision and art.

The primary objective of this project was to investigate how GANs may be used to generate anime characters and Pokémon, giving people a platform to design their own distinctive characters. By harnessing the power of GANs, we aim to automate and streamline the character design process, enabling users to have greater creative control and flexibility for customizations.

The project is divided into two main components: anime character generation and Pokémon generation. For anime character generation, a GAN architecture is trained on a large set of 2D anime character images, encompassing a wide range of styles, aesthetics, and visual characteristics. By identifying the underlying patterns and traits present in the training data, the GAN learns from this dataset and creates new concepts for anime characters.

Similarly, a different GAN is trained on a collection of all the existing Pokémon photos from all the generations. The GAN can create new Pokémon concepts because it learns the rich intricacies and distinctive characteristics of each Pokémon species. This strategy guarantees that the created Pokémon have distinctive looks, including a range of body forms, color palettes, and distinguishing traits.

To further enhance the user experience and facilitate customization, a separate deep learning model is integrated into the project. This model predicts the top types of each generated Pokémon, providing users with an additional layer of control over their creations. The deep learning model analyzes the visual attributes of the Pokémon and determines the most suitable types based on similarities to existing Pokémon types. This feature empowers users to align their generated Pokémon with specific attributes, such as grass, fire, water, electric, and more (16 total).

Users are given the chance to construct customized characters and Pokémon that match their tastes by integrating the capabilities of GANs with their choice of preferred types. The project seeks to meet the demands of users from a variety of backgrounds, from artists to enthusiasts, who can easily explore different character designs and produce original concepts.

The application of GANs in character generation not only simplifies the design process but also presents opportunities for creativity and exploration. By automating certain aspects of character design, artists can focus more on the ideation and storytelling aspects, allowing for deeper immersion in the narrative.

Overall, this project introduces an innovative approach to character design through the utilization of GANs. The combination of anime character generation and Pokémon generation, coupled with the ability to select types, empowers users to create unique and visually captivating characters and creatures. The initiative offers a platform for fans to develop their creative potential in addition to meeting the demands of artists and designers. The possibilities for character creation and customization are endless given the rapid advances in machine learning and AI, and this project acts as an introduction towards uncovering those possibilities.

## Related Work
The application of GANs in character generation has garnered significant attention in recent years, especially with major game studios relying on them to generate character designs. Extensive work has been done in this field to improve the quality and diversity of generated characters.

One common approach involves employing deep convolutional neural networks (CNNs) as the base architecture for the GAN model, such as the one proposed by Radford et al.[1] These CNNs demonstrate the ability to capture intricate details and produce visually appealing characters. Since simple GANs generate seemingly random images,  researchers have investigated the integration of conditional GANs, enabling control over specific attributes or styles of the generated characters, first demonstrated by Mirza & Osindero et al. [2] This has paved the way for interactive character generation, where users can select desired traits for the generated character. Moreover, efforts have been made to leverage unsupervised learning methods, such as Variational Autoencoders (VAEs), in combination with GANs to generate more diverse and coherent character representations.

Anime characters in particular have distinctive yet separate styles where each series/artist may have different style yet the artwork is classified as “anime style”. In order to replicate this delicate style, features have to be selected carefully. An anime character GAN by Yanghua et al. [3] makes use of a GAN model based on DRAGAN which consists of the ability to select features while providing coherent images.

## Methodology
_A. Data Collection_ <br>
**Anime Character Dataset:** A dataset from Kaggle named “Anime Faces” is used. It is a diverse dataset consisting of 21551 anime character images that were collected from online sources. All the images have dimensions of 64*64 [4]. <br>
Fig. 1. shown below showcases some of the images from the dataset.
![Anime Images Dataset](https://github.com/RishabhSpark/Pokemon-Generator-GANs-with-Type-Customization/blob/main/images/anime%20dataset.png)

**Pokémon Images Dataset:** A comprehensive dataset of existing Pokemon images is extracted from an online Pokémon database. The dataset should cover all of the existing Pokemon species, capturing their unique designs, body shapes, colors, and features. All the images have dimensions of 120*120 pixels.
Figure 2 displays some examples of the entries from the Pokemon images dataset generated.
![Pokemon Images Dataset](https://github.com/RishabhSpark/Pokemon-Generator-GANs-with-Type-Customization/blob/main/images/pokemon%20dataset.png)

**Pokémon Stats Dataset:** A comprehensive dataset of all of the existing Pokemon stats is extracted from an online Pokémon database.

_B. Preprocessing_ <br>
**Anime Character Generation:** The collected anime character images are preprocessed to ensure consistency and improve training performance. This involves normalizing pixel values.

**Pokémon Generation:** Similar preprocessing steps are applied to the collected Pokemon images. The images are resized to a standardized resolution, normalized, and converted to a compatible format.

**Pokemon Types Modeling:** The dataset is then preprocessed eliminating the insignificant stats like HP. attack, def, etc. This leaves the Pokemon names and types as columns. There are a total of 18 different types, so the types are preprocessed by the one-hot encoding technique.

_C. Model Creation and Training_ <br>
**Anime Character Generation:** A Deep Convolutional Generative Adversarial Network (DCGAN) is implemented and trained on the preprocessed anime character dataset. The generator network learns to generate new anime character images, while the discriminator network distinguishes between real and generated images. Initially, the discriminator is trained, and then the generator and discriminator are both trained simultaneously, optimizing their respective objectives using techniques like adversarial loss and gradient descent.<br>
The model was trained for a total of 150 epochs.

Figure 3. shows the flowchart of the entire model of Anime Character Generation.

![Flowchart Anime](https://github.com/RishabhSpark/Pokemon-Generator-GANs-with-Type-Customization/blob/main/images/Anime%20Model.png)

**Pokémon Generation:** Another DCGAN for generating images is implemented and trained on the preprocessed Pokemon dataset. The generator learns to produce novel Pokemon images, while the discriminator aims to differentiate between real and generated Pokemon images. The training process involves iteratively updating the generator and discriminator parameters to improve the quality and diversity of the generated Pokemon.

But, in addition to this, the Pokemon Generation model has some type of feature for customization based on user input.

The model was trained for a total of 500 epochs, considering the resources available.

**Type Prediction Model Training:** To facilitate the Pokemon customization, a separate deep learning model, a Convolutional Neural Network (CNN), is trained to predict the top types of each generated Pokemon. This model takes the generated Pokemon images as input and outputs the probability of all the types based on learned patterns from the existing Pokemon types.

The CNN model is constructed, consisting of convolutional layers, pooling layers, fully connected layers, and an output layer with softmax activation for type prediction. The model also uses dropout layers at several stages. The model is trained using techniques like the Adam optimizer and categorical cross-entropy loss, optimizing the parameters based on the labeled dataset.

_D. Integration_ <br>
The trained anime character GAN, Pokemon GAN, and type prediction model are then finally integrated by exporting the models and combining them together.
The interface provides options for users to generate anime characters and Pokemon based on their preferences. For Pokemon generation, users can generate a new Pokemon every time, and it assigns two types to the Pokemon that are generated.

The type prediction model assists users in selecting the most suitable types for their generated Pokemon based on visual similarities to existing Pokemon types. Users can choose from a list of predicted types or manually assign types to their Pokemon as well.

_E. Evaluation_ <br>
The generated anime characters and Pokemon are evaluated for their visual quality, diversity, and adherence to desired characteristics.
                    
But, due to resource limitations, things such as accuracy and loss.

## Results
_A. Anime Character Generation_ <br>
The training of the anime character GAN was carried out over multiple epochs to capture the progression and improvement of the generated anime character concepts. Figure 4 showcases the generated anime characters at different epochs, namely Epoch 1, Epoch 30, Epoch 60, Epoch 90, Epoch 120, and Epoch 150.

![Anime Results](https://github.com/RishabhSpark/Pokemon-Generator-GANs-with-Type-Customization/blob/main/images/Anime%20results.png)

At Epoch 1, the generated images have a very random noise pattern but still exhibit some level of resemblance to the training dataset but lack fine details and diversity. As the training progresses, the GAN learns to capture more intricate features and diverse styles, resulting in anime characters that are visually appealing and unique. By Epoch 150, the generated anime characters demonstrate significant improvement, with various hairstyles, facial features, and clothing styles.

_B. Pokemon Generation_ <br>
The Pokemon GAN was trained over a larger number of epochs to capture the complexity and diversity of the Pokemon species. Figure 5 showcases the generated Pokemon images at different epochs, including Epoch 1, Epoch 100, Epoch 200, Epoch 300, Epoch 400, and Epoch 500.

![Pokemon results](https://github.com/RishabhSpark/Pokemon-Generator-GANs-with-Type-Customization/blob/main/images/Pokemon%20results.png)

In the initial epochs, such as Epoch 1, the generated Pokemon images exhibit random patterns and lack distinct characteristics. However, as the training progresses, the GAN learns to produce more recognizable and diverse Pokemon designs. By Epoch 500, the generated Pokemon encompassed a wide range of body shapes, color schemes, and unique features, closely resembling the existing Pokemon species. Yet still not fully generating finer details because of a lack of runs. The results get better over time.

_C. Pokemon Type Labels_
The labeling section of the project is dedicated to predicting the top types of each generated Pokemon. The labeling section utilizes a trained deep-learning CNN model to analyze the visual attributes of the generated Pokemon and predict the most suitable types. This assists users in aligning their generated Pokemon with specific characteristics and attributes they desire, such as fire, water, electric, and more.

A matrix of all the types is generated, and the top values are picked from it to be displayed. Figure 6 provides a screenshot of the labeling section, where the generated Pokemon image is displayed alongside the predicted types based on visual analysis.

![Labeling](https://github.com/RishabhSpark/Pokemon-Generator-GANs-with-Type-Customization/blob/main/images/labeling.png)

_D. Integrated Pokemon Project_
The integrated project offers a user to create personalized anime characters and Pokemon and visualize the predicted types.

Figure 7 shows the final output generated by the user. It consists of a Pokemon with the most probably two types.

![Final Results](https://github.com/RishabhSpark/Pokemon-Generator-GANs-with-Type-Customization/blob/main/images/final.png)

## Conclusion
This project demonstrated the successful application of Generative Adversarial Networks (GANs) in generating anime characters and Pokemon, offering users a platform to unleash their creativity and customize their creations based on specific attributes. By leveraging the power of GANs, the project facilitated the automated generation of visually diverse and appealing character concepts.

The anime character GAN showcased substantial progress throughout the training process, evolving from rudimentary resemblances to the training dataset to intricate and unique character designs. As the training advanced, the generated anime characters exhibited various hairstyles, facial features, and clothing styles, enabling users to explore a wide range of possibilities.

Similarly, the Pokemon GAN displayed significant improvement over numerous epochs, capturing the complexity and diversity of existing Pokemon species. The generated Pokemon images evolved from random patterns to recognizable designs, encompassing a plethora of body shapes, color schemes, and defining features. This allowed users to create their own Pokemon with distinct visual characteristics.

The integration of a type prediction model further enhanced the project's capabilities by assisting users in selecting the most suitable types for their generated Pokemon. By analyzing the visual attributes of the generated Pokemon, the model predicted the top types based on similarities to existing Pokemon types. This feature enabled users to align their creations with specific attributes, such as fire, water, electric, and more, providing a personalized touch to their Pokemon designs.

The seamless integration of the different components resulted in a user-friendly interface that empowered artists, designers, and enthusiasts to explore various character designs and unleash their creative potential. The project not only streamlined the character design process but also provided a platform for users to visualize and customize their creations, bringing their imaginations to life.

The success of this project highlights the potential of GANs in automating and revolutionizing the character design domain. By leveraging machine learning techniques, such as GANs and deep learning models, artists can focus more on ideation and storytelling aspects, while the AI systems handle the generation and customization processes. This synergy between human creativity and artificial intelligence opens new avenues for character design and customization in the realms of video games, anime, and beyond.

As technology continues to advance, the possibilities for character design and customization using GANs and AI-driven systems are boundless. Future research in this field can explore further improvements in GAN architectures, fine-tuning type prediction models, and expanding the dataset for more diverse and realistic character generation. The integration of user feedback and preferences can also contribute to enhancing the overall user experience and refining the generated character concepts.

## Future Work
Optimization of Training Process: To address the limitation of training time, future work can focus on optimizing the training process of the GAN models. This can involve exploring advanced training techniques such as progressive growing or network architecture modifications to accelerate convergence and improve the efficiency of training. Additionally, hardware upgrades or utilization of cloud-based computing resources can significantly reduce training time and facilitate experimentation with larger datasets.

Extended Training Duration: Considering that 150 or 500 epochs may not be sufficient to generate optimal results, future work should investigate training the GAN models for a more extended period. Increasing the number of epochs beyond the initial benchmarks can allow the models to capture more nuanced details, improve diversity, and achieve better convergence. Experimentation with longer training durations can lead to the generation of even more realistic and visually appealing anime characters and Pokemon.

Multi-modal Approach for Type Prediction: To address the limitation of relying solely on visual similarity for predicting Pokemon types, future work can explore a multi-modal approach. Integration of additional data sources, such as textual descriptions, gameplay attributes, and historical Pokemon data, can provide a more comprehensive understanding of Pokemon types. By incorporating multiple modalities, such as combining visual analysis with textual information, the accuracy and reliability of the type prediction model can be significantly improved.

Incorporating Diffuser Model: By incorporating a diffuser model, the generated anime characters and Pokemon can exhibit smoother transitions and finer details, resulting in more visually appealing and refined designs. The diffuser model offers the ability to regulate the trade-off between exploration and exploitation during the generation process, addressing mode collapse issues and introducing variability in the generated images. Users can also benefit from more fine-grained control over the generated results, actively shaping and customizing the character concepts according to their preferences. The integration of the diffuser model opens up possibilities for novel applications such as style transfer and cross-domain generation, allowing users to blend different art styles and merge characteristics from multiple Pokemon types.

## References
[1] Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015. <br>
[2] Mehdi Mirza and Simon Osindero. Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784, 2014.<br>
[3] Jin, Yanghua & Zhang, Jiakai & Li, Minjun & Tian, Yingtao & Zhu, Huachun & Fang, Zhihao. (2017). Towards the Automatic Anime Characters Creation with Generative Adversarial Networks. <br>
[4] https://www.kaggle.com/datasets/soumikrakshit/anime-faces


## Links
DCGAN Pokemon (Notebook) - https://www.kaggle.com/code/rishabhspark/dcgans-pokemon/notebook <br>
Pokemon Type Labelling (Notebook) - https://www.kaggle.com/code/rishabhspark/labeling/notebook <br>
Pokemon Image and Types dataset - https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types <br>
DCGAN Anime (Notebook) - https://www.kaggle.com/code/rishabhspark/dcgan-anime/notebook <br>
Anime Faces Dataset - https://www.kaggle.com/datasets/soumikrakshit/anime-faces <br>
Pokemon data webscraped from https://pokemondb.net/
