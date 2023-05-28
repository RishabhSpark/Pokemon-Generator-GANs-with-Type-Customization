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

**Pokémon Images Dataset:** A comprehensive dataset of existing Pokemon images is extracted from an online Pokémon database. The dataset should cover all of the existing Pokemon species, capturing their unique designs, body shapes, colors, and features. All the images have dimensions of 120*120 pixels.
Figure 2 displays some examples of the entries from the Pokemon images dataset generated.

**Pokémon Stats Dataset:** A comprehensive dataset of all of the existing Pokemon stats is extracted from an online Pokémon database.

_B. Preprocessing_
**Anime Character Generation:** The collected anime character images are preprocessed to ensure consistency and improve training performance. This involves normalizing pixel values.

**Pokémon Generation:** Similar preprocessing steps are applied to the collected Pokemon images. The images are resized to a standardized resolution, normalized, and converted to a compatible format.

**Pokemon Types Modeling:** The dataset is then preprocessed eliminating the insignificant stats like HP. attack, def, etc. This leaves the Pokemon names and types as columns. There are a total of 18 different types, so the types are preprocessed by the one-hot encoding technique.

## Links
DCGAN Pokemon (Notebook) - https://www.kaggle.com/code/rishabhspark/dcgans-pokemon/notebook <br>
Pokemon Type Labelling (Notebook) - https://www.kaggle.com/code/rishabhspark/labeling/notebook <br>
Pokemon Image and Types dataset - https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types <br>
DCGAN Anime (Notebook) - https://www.kaggle.com/code/rishabhspark/dcgan-anime/notebook <br>
Anime Faces Dataset - https://www.kaggle.com/datasets/soumikrakshit/anime-faces <br>
Pokemon data webscraped from https://pokemondb.net/
