# Ischemic-Stroke-Classification

One of the main causes of death and permanent disability is ischemic stroke, for which
prompt and precise diagnosis is essential to successful treatment. This study introduces a novel dual-stream
deep learning framework for ischemic stroke classification using Computed Tomography (CT) images,
specifically addressing challenges in accuracy, computational efficiency, and clinical interpretability. Three
significant innovations are included in the suggested architecture: (1) a hybrid Dual Attention Mechanism
that combines Dynamic Routing and Cross-Attention for improved region-specific feature discrimination;
(2) a Multi-Scale Feature Extraction Module with parallel convolutional pathways that captures both
contextual and fine-grained features; and (3) an Adaptive Random Vector Functional Link layer that
significantly reduces training time while maintaining high classification performance. When tested on a
single-center dataset, the model achieves state-of-the-art classification accuracy of 98.83% across normal,
acute and chronic stroke categories. We demonstrate the strong generalization capabilities of the proposed
framework by achieving 92.42% accuracy on a diverse, multi-center dataset of 7,842 CT images. The
integration of explainable Artificial Intelligence tools improves clinical trustworthiness by offering clear
insight into the model’s decision-making process. These outcomes demonstrate the model’s potential for
use in actual clinical settings for quick and accurate stroke diagnosis, along with its interpretability and
computational efficiency.
