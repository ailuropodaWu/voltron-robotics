from torchvision.io import read_image
from voltron import instantiate_extractor, load

# Load a frozen Voltron (V-Cond) model & configure a vector extractor
vcond, preprocess = load("v-cond", device="cuda", freeze=True)
vector_extractor = instantiate_extractor(vcond)()

# Obtain & Preprocess an image =>> can be from a dataset, or camera on a robot, etc.
#   => Feel free to add any language if you have it (Voltron models work either way!)
img = preprocess(read_image("examples/verification/img/peel-carrot-initial.png"))[None, ...].to("cuda")
lang = ["peeling a carrot"]

# Extract both multimodal AND vision-only embeddings!
multimodal_embeddings = vcond(img, lang, mode="multimodal")
visual_embeddings = vcond(img, mode="visual")

# Use the `vector_extractor` to output dense vector representations for downstream applications!
#   => Pass this representation to model of your choice (object detector, control policy, etc.)
representation = vector_extractor(multimodal_embeddings)