from torchvision import transforms

MushroomTransform = transforms.Compose([transforms.Resize([256, 256]),
                                        transforms.ToTensor()])