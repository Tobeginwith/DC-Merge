####################### 8 Vision Benchmark #######################

cars_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'i love my {c}!',
    lambda c: f'a photo of my dirty {c}.',
    lambda c: f'a photo of my clean {c}.',
    lambda c: f'a photo of my new {c}.',
    lambda c: f'a photo of my old {c}.',
]

dtd_template = [
    lambda c: f'a photo of a {c} texture.',
    lambda c: f'a photo of a {c} pattern.',
    lambda c: f'a photo of a {c} thing.',
    lambda c: f'a photo of a {c} object.',
    lambda c: f'a photo of the {c} texture.',
    lambda c: f'a photo of the {c} pattern.',
    lambda c: f'a photo of the {c} thing.',
    lambda c: f'a photo of the {c} object.',
]

gtsrb_template = [
    lambda c: f'a zoomed in photo of a "{c}" traffic sign.',
    lambda c: f'a centered photo of a "{c}" traffic sign.',
    lambda c: f'a close up photo of a "{c}" traffic sign.',
]

mnist_template = [
    lambda c: f'a photo of the number: "{c}".',
]

eurosat_template = [
    lambda c: f'a centered satellite photo of {c}.',
    lambda c: f'a centered satellite photo of a {c}.',
    lambda c: f'a centered satellite photo of the {c}.',
]
resisc45_template = [
    lambda c: f'satellite imagery of {c}.',
    lambda c: f'aerial imagery of {c}.',
    lambda c: f'satellite photo of {c}.',
    lambda c: f'aerial photo of {c}.',
    lambda c: f'satellite view of {c}.',
    lambda c: f'aerial view of {c}.',
    lambda c: f'satellite imagery of a {c}.',
    lambda c: f'aerial imagery of a {c}.',
    lambda c: f'satellite photo of a {c}.',
    lambda c: f'aerial photo of a {c}.',
    lambda c: f'satellite view of a {c}.',
    lambda c: f'aerial view of a {c}.',
    lambda c: f'satellite imagery of the {c}.',
    lambda c: f'aerial imagery of the {c}.',
    lambda c: f'satellite photo of the {c}.',
    lambda c: f'aerial photo of the {c}.',
    lambda c: f'satellite view of the {c}.',
    lambda c: f'aerial view of the {c}.',
]

sun397_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
]

svhn_template = [
    lambda c: f'a photo of the number: "{c}".',
]

flowers102_template = [
    lambda c: f"a photo of a {c}, a type of flower.",
]

oxfordpets_template = [
    lambda c: f"a photo of a {c}, a type of pet.",
]

cifar100_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a blurry photo of a {c}.",
    lambda c: f"a black and white photo of a {c}.",
    lambda c: f"a low contrast photo of a {c}.",
    lambda c: f"a high contrast photo of a {c}.",
    lambda c: f"a bad photo of a {c}.",
    lambda c: f"a good photo of a {c}.",
    lambda c: f"a photo of a small {c}.",
    lambda c: f"a photo of a big {c}.",
    lambda c: f"a photo of the {c}.",
    lambda c: f"a blurry photo of the {c}.",
    lambda c: f"a black and white photo of the {c}.",
    lambda c: f"a low contrast photo of the {c}.",
    lambda c: f"a high contrast photo of the {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a good photo of the {c}.",
    lambda c: f"a photo of the small {c}.",
    lambda c: f"a photo of the big {c}.",
]

cifar10_stl10_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a blurry photo of a {c}.",
    lambda c: f"a black and white photo of a {c}.",
    lambda c: f"a low contrast photo of a {c}.",
    lambda c: f"a high contrast photo of a {c}.",
    lambda c: f"a bad photo of a {c}.",
    lambda c: f"a good photo of a {c}.",
    lambda c: f"a photo of a small {c}.",
    lambda c: f"a photo of a big {c}.",
    lambda c: f"a photo of the {c}.",
    lambda c: f"a blurry photo of the {c}.",
    lambda c: f"a black and white photo of the {c}.",
    lambda c: f"a low contrast photo of the {c}.",
    lambda c: f"a high contrast photo of the {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a good photo of the {c}.",
    lambda c: f"a photo of the small {c}.",
    lambda c: f"a photo of the big {c}.",
]

fer2013_template = [
    lambda c: f"a photo of a {c} looking face.",
    lambda c: f"a photo of a face showing the emotion: {c}.",
    lambda c: f"a photo of a face looking {c}.",
    lambda c: f"a face that looks {c}.",
    lambda c: f"they look {c}.",
    lambda c: f"look at how {c} they are.",
]

sst2_template = [
    lambda c: f"a {c} review of a movie.",
]

fashionmnist_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of the {c}.",
]

dataset_to_template = {
    'stanford_cars': cars_template,
    'dtd': dtd_template,
    'eurosat': eurosat_template,
    'gtsrb': gtsrb_template,
    'mnist': mnist_template,
    'resisc45': resisc45_template,
    'sun397': sun397_template,
    'svhn': svhn_template,
    'oxfordpets': oxfordpets_template,
    'cifar100': cifar100_template, 
    'flowers102': flowers102_template, 
    'stl10': cifar10_stl10_template,
    'sst2': sst2_template,
    'fashionmnist': fashionmnist_template, 
    'fer2013': fer2013_template,
    'cifar10': cifar10_stl10_template,
}


def get_templates(dataset_name):
    if dataset_name.endswith('Val'):
        return get_templates(dataset_name.replace('Val', ''))
    assert dataset_name in dataset_to_template, f'Unsupported dataset: {dataset_name}'
    return dataset_to_template[dataset_name]