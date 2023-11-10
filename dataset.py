import torch
from transformers import AutoTokenizer, pipeline
animal_list = [
    "Aardvark", "Abyssinian", "Afghan Hound", "African Bullfrog", "African Elephant",
    "African Forest Elephant", "African Grey Parrot", "African Leopard", "African Rock Python", "Airedale Terrier",
    "Akita", "Albatross", "Alligator", "Alpaca", "American Bulldog",
    "American Cocker Spaniel", "American Eskimo Dog", "American Foxhound", "American Pit Bull Terrier", "Amur Leopard",
    "Angelfish", "Ant", "Anteater", "Antelope", "Appenzeller Dog",
    "Arctic Fox", "Arctic Hare", "Arctic Wolf", "Armadillo", "Asian Elephant",
    "Asian Giant Hornet", "Asian Palm Civet", "Aye Aye", "Bactrian Camel", "Balinese",
    "Banded Palm Civet", "Bandicoot", "Barb", "Barn Owl", "Barnacle",
    "Barracuda", "Barramundi Fish", "Basenji Dog", "Basking Shark", "Basset Fauve de Bretagne",
    "Basset Hound", "Bat", "Bavarian Mountain Hound", "Beagle", "Bear",
    "Beaver", "Bedlington Terrier", "Beetle", "Bengal Tiger", "Bernese Mountain Dog",
    "Bichon Frise", "Binturong", "Bird", "Birds Of Paradise", "Birman",
    "Bison", "Black Marlin", "Black Rhinoceros", "Black Russian Terrier", "Bloodhound",
    "Booby", "Bolognese Dog", "Bombay", "Bongo", "Bonito Fish",
    "Bonobo", "Bowerbird", "Boxer Dog", "Boykin Spaniel", "Brazilian Terrier",
    "Brown Bear", "Budgerigar", "Buffalo", "Bulldog", "Bull Shark",
    "Bull Terrier", "Bulldog", "Bullfrog", "Bumblebee", "Burmese",
    "Burmilla", "Butterfly", "Butterfly Fish", "Buzzard", "Caecilian",
    "Cairn Terrier", "Calmar", "Camel", "Capybara", "Caracal",
    "Carpenter Ant", "Cassowary", "Cat", "Caterpillar", "Catfish",
    "Cavalier King Charles Spaniel", "Centipede", "Chameleon", "Chamois", "Cheetah",
    "Chesapeake Bay Retriever", "Chicken", "Chickadee", "Chihuahua", "Chimaera",
    "Chimpanzee", "Chinook", "Chinchilla", "Chinese Crested Dog", "Chipmunk",
    "Chondrichthyes", "Chrysanthemum", "Chupacabra", "Cichlid", "Clam",
    "Clownfish", "Clumber Spaniel", "Coati", "Cockroach", "Collared Peccary",
    "Collie", "Common Buzzard", "Common Frog", "Common Loon", "Common Toad",
    "Coral", "Cottontop Tamarin", "Cougar", "Cow", "Coyote",
    "Crab", "Crane", "Crawfish", "Crocodile", "Cuscus",
    "Cuttlefish", "Dachshund", "Dalmatian", "Darwins Frog", "Dart Frog",
    "Deer", "Desert Tortoise", "Deutsche Bracke", "Dhole", "Dingo",
    "Discus", "Doberman Pinscher", "Dogfish", "Dolphin", "Donkey",
    "Dorado", "Dragonfly", "Drever", "Duck", "Dugong",
    "Dunker", "Dusky Grouse", "Dwarf Crocodile", "Eagle", "Earwig",
    "Eastern Gorilla", "Eastern Towhee", "Echidna", "Edible Frog", "Egyptian Mau",
    "Electric Eel", "Elephant", "Elephant Seal", "Elephant Shrew", "Emperor Penguin",
    "Emu", "English Cocker Spaniel", "English Foxhound", "English Setter", "Ermine",
    "Eurasian Magpie", "Falcon", "False Killer Whale", "Fennec Fox", "Ferret",
    "Field Spaniel", "Fin Whale", "Finch", "Fire-Bellied Toad", "Fish",
    "Fishing Cat", "Flamingo", "Flat-Coated Retriever", "Flounder", "Fly",
    "Flying Fish", "Fogfish", "Forest Cobra", "French Bulldog", "Frigatebird",
    "Frilled Lizard", "Fur Seal", "Galápagos Tortoise", "Galah", "Gar",
    "Gecko", "Gentoo Penguin", "Gerbil", "German Pinscher", "German Shepherd",
    "Gharial", "Giant African Land Snail", "Giant Clam", "Giant Panda", "Giant Schnauzer",
    "Gibbon", "Giraffe", "Glass Lizard", "Glow Worm", "Goanna",
    "Golden Lion Tamarin", "Golden Oriole", "Golden Retriever", "Goldeneye", "Goldfish",
    "Goosander", "Goose", "Gopher Snake", "Gorilla", "Goshawk",
    "Grasshopper", "Great Dane", "Great White Shark", "Greater Swiss Mountain Dog", "Green Bee-Eater",
    "Green Iguana", "Greenland Dog", "Grey Mouse Lemur", "Grey Reef Shark", "Grey Seal",
    "Greyhound", "Grizzly Bear", "Grouse", "Guanaco", "Guinea Fowl",
    "Guinea Pig", "Guppy", "Hammerhead Shark", "Hamster", "Hare",
    "Harrier", "Havanese", "Hawk", "Hedgehog", "Hermit Crab",
    "Heron", "Highland Cattle", "Himalayan", "Hippopotamus", "Honey Bee",
    "Horn Shark", "Horned Frog", "Horse", "Horseshoe Crab", "Howler Monkey",
    "Human", "Humboldt Penguin", "Hummingbird", "Hungarian Puli", "Hungarian Vizsla",
    "Ibis", "Ibizan Hound", "Iguana", "Impala", "Indian Elephant",
    "Indian Palm Squirrel", "Indian Rhinoceros", "Indian Star Tortoise", "Indochinese Tiger", "Indri",
    "Insect", "Irish Setter", "Irish Wolfhound", "Irukandji Jellyfish", "Italian Greyhound",
    "Jack Russell", "Jackal", "Jaguar", "Japanese Chin", "Japanese Macaque",
    "Japanese Spitz", "Japanese Terrier", "Javan Rhinoceros", "Jellyfish", "Jerboa",
    "Kangaroo", "Keel-Billed Toucan", "Killer Whale", "King Crab", "King Penguin",
    "Kingfisher", "Kinkajou", "Kissing Gourami", "Kiwi", "Koala",
    "Komodo Dragon", "Kudu", "Labradoodle", "Labrador Retriever", "Ladybug",
    "Lagotto Romagnolo", "Lamb", "Lancaster Bomber", "Leaf-Tailed Gecko", "Lemming",
    "Lemur", "Leopard", "Leopard Cat", "Leopard Gecko", "Lesser Panda",
    "Lesser Sulphur-Crested Cockatoo", "Liger", "Liger Liger", "Lilac Breasted Roller", "Llama",
    "Lobster", "Long-Eared Owl", "Lynx", "Macaroni Penguin", "Macaw",
    "Magellanic Penguin", "Magpie", "Maine Coon", "Malayan Civet", "Malayan Tiger",
    "Maltese", "Manatee", "Mandarin Duck", "Mandrill", "Markhor",
    "Marmoset", "Marmot", "Marsh Frog", "Mastiff", "Mayfly",
    "Meerkat", "Mexican Hairless Dog", "Midge", "Millipede", "Minke Whale",
    "Mole", "Molly", "Mongrel", "Monkfish", "Monroe",
    "Monte", "Moorhen", "Moose", "Moray Eel", "Moth",
    "Mountain Gorilla", "Mountain Lion", "Mouse", "Mule", "Numbat",
    "Nudibranch", "Nalolo", "Neddicky", "Nematode", "Newfypoo",
    "Nicator", "Ocelot", "Octopus", "Ostrich", "Orangutan",
    "Okapi", "Oxpecker", "Oystercatcher", "Oarfish", "Obeliscus",
    "Olive Baboon", "Panther", "Pangolin", "Puma", "Python",
    "Parrot", "Penguin", "Porcupine", "Polar Bear", "Platypus",
    "Praying Mantis", "Puffin", "Pika", "Paddlefish", "Prawn",
    "Peacock", "Pelican", "Pigeon", "Prawn", "Pronghorn",
    "Quail", "Quokka", "Quoll", "Queen Angelfish", "Queen Triggerfish",
    "Quetzal", "Quillfish", "Quiver Tree Frog", "Quoll", "Quahog",
    "Rabbit", "Raccoon", "Rattlesnake", "Red Panda", "Reindeer",
    "Rhea", "Rhino", "Ring-tailed Lemur", "River Dolphin", "Robin",
    "Rockhopper Penguin", "Rooster", "Rottweiler (dog breed)", "Ruby-throated Hummingbird", "Ruddy Duck",
    "Rufous Hornbill", "Ruffed Lemur", "Russian Blue (cat breed)", "Saola", "Salamander",
    "Sambar Deer", "Sand Dollar", "Sand Eel", "Sandpiper", "Sawfish",
    "Scallop", "Scarlet Ibis", "Scorpion", "Sea Anemone", "Sea Dragon",
    "Sea Lion", "Sea Otter", "Sea Slug", "Sea Squirt", "Sea Turtle",
    "Sea Urchin", "Seahorse", "Serval", "Shark", "Shepherd Dog",
    "Shih Tzu (dog breed)", "Shrimp", "Siamese (cat breed)", "Siberian Husky (dog breed)", "Silver Dollar Fish",
    "Skunk", "Sloth", "Snail", "Snake", "Snow Leopard",
    "Snowshoe (cat breed)", "Somali (cat breed)", "Sparrow", "Spider", "Spoonbill",
    "Squid", "Squirrel", "Starfish", "Star-nosed Mole", "Stingray",
    "Stoat", "Stork", "Sumatran Tiger", "Sun Bear", "Sunfish",
    "Swan", "Swordfish", "Tamarin", "Tang Fish", "Tapir",
    "Tarsier", "Tasmanian Devil", "Tawny Owl", "Teddy Bear Dog", "Termite",
    "Tetra", "Thorny Devil", "Tiger", "Tiger Salamander", "Tiger Shark",
    "Tortoise", "Toucan", "Tree Frog", "Tree Kangaroo", "Tropicbird",
    "Trout", "Tuna", "Turkey", "Turkish Angora (cat breed)", "Turtle",
    "Uakari", "Uguisu (Japanese Bush Warbler)", "Umbrellabird", "Unicornfish", "Urial",
    "Urutu (snake)", "Utahraptor", "Vampire Bat", "Vampire Squid", "Vervet Monkey",
    "Vicuña", "Viperfish", "Vizsla (dog breed)", "Volpino Italiano (dog breed)", "Vulture",
    "Wallaby", "Walrus", "Warthog", "Wasp", "Water Buffalo",
    "Water Dragon", "Water Vole", "Weasel", "Weaver Bird", "Weimaraner (dog breed)",
    "Welsh Corgi (dog breed)", "West Highland White Terrier (dog breed)", "Whale Shark", "Whippet (dog breed)",
    "White Rhinoceros", "White Tiger", "Whooping Crane", "Wigeon", "Wild Boar",
    "Wildebeest", "Wire Fox Terrier (dog breed)", "Wolf", "Wolverine", "Wombat",
    "Woodpecker", "Worm", "Wrasse", "Wren", "Wombat",  "Woodpecker",  "Worm", "Wrasse", "Wren",
    "X-ray Tetra", "X-ray Fish (Pristella maxillaris)", "Xenops", "Xoloitzcuintli (Mexican Hairless Dog)", "X-Ray Tetra", "X-Ray Fish",
    "Yak", "Yellow-eyed Penguin", "Yellowfin Tuna", "Yorkie Poo (dog breed)", "Yorkshire Terrier (dog breed)",
    "Zebra", "Zebu", "Zorilla (African polecat)", "Zebroid (generic term for Zebra hybrids)", "Zebu (cattle breed)"
]

len(animal_list)

bird_list = [
    "African Grey Parrot", "Albatross", "American Crow", "American Goldfinch", "American Kestrel",
    "American Robin", "Andean Condor", "Atlantic Puffin", "Bald Eagle", "Baltimore Oriole",
    "Bananaquit", "Barn Owl", "Barn Swallow", "Barred Owl", "Bateleur Eagle", "Bearded Vulture",
    "Belted Kingfisher", "Black Skimmer", "Black Swan", "Blue Jay", "Blue Tit", "Bohemian Waxwing",
    "Brewer's Blackbird", "Brown Pelican", "Budgerigar", "Cactus Wren", "California Condor",
    "Canada Goose", "Canary", "Cape May Warbler", "Cardinal", "Carolina Wren", "Cassowary",
    "Cedar Waxwing", "Cattle Egret", "Chaffinch", "Chipping Sparrow", "Cockatiel", "Cockatoo",
    "Collared Dove", "Common Grackle", "Common Loon", "Common Nighthawk", "Common Pheasant",
    "Common Raven", "Common Redpoll", "Coot", "Cormorant", "Crane", "Crested Caracara",
    "Crimson Rosella", "Crow", "Cuckoo", "Currawong", "Dodo", "Downy Woodpecker", "Dusky Grouse",
    "Eagle", "Eastern Bluebird", "Eastern Screech Owl", "Eastern Towhee", "Egret", "Emu",
    "Eurasian Magpie", "European Robin", "Falcon", "Finch", "Flamingo", "Frigatebird", "Galah",
    "Gentoo Penguin", "Gila Woodpecker", "Goldcrest", "Golden Eagle", "Goosander", "Gouldian Finch",
    "Great Blue Heron", "Great Crested Flycatcher", "Great Egret", "Greater Prairie Chicken",
    "Green Heron", "Grey Crowned Crane", "Grey Heron", "Grosbeak", "Gull", "Gyrfalcon",
    "Harpy Eagle", "Harrier", "Harris's Hawk", "Hawk", "Helmeted Guineafowl", "Hoopoe", "Horned Owl",
    "House Sparrow", "House Wren", "Humboldt Penguin", "Hyacinth Macaw", "Ibex", "Inca Tern",
    "Indigo Bunting", "Jacana", "Jackdaw", "Keel-billed Toucan", "Killdeer", "Kingfisher", "Kiwi",
    "Kookaburra", "Lark", "Limpkin", "Lory", "Lorikeet", "Loon", "Lorikeet", "Lovebird", "Macaw",
    "MacGillivray's Warbler", "Magellanic Penguin", "Magpie", "Mallard", "Mandarin Duck",
    "Mistle Thrush", "Mockingbird", "Moorhen", "Motmot", "Mynah Bird", "Nighthawk", "Northern Cardinal",
    "Northern Mockingbird", "Northern Saw-whet Owl", "Nutcracker", "Osprey", "Ostrich", "Ovenbird",
    "Owl", "Painted Bunting", "Palm Cockatoo", "Parakeet", "Parrot", "Partridge", "Peacock", "Pelican",
    "Penguin", "Peregrine Falcon", "Pheasant", "Pigeon", "Pintail Duck", "Piping Plover", "Plover",
    "Potoo", "Puffin", "Quail", "Quetzal", "Raven", "Red Kite", "Red-shouldered Hawk", "Red-tailed Hawk",
    "Red-winged Blackbird", "Rhea", "Roadrunner", "Robin", "Rock Dove", "Roseate Spoonbill", "Ross's Gull",
    "Ruffed Grouse", "Rufous Hummingbird", "Sabine's Gull", "Sage Grouse", "Sandhill Crane", "Sandpiper",
    "Sanderling", "Scarlet Ibis", "Screech Owl", "Secretary Bird", "Secretarybird", "Shoveler Duck", "Siskin",
    "Snipe", "Snow Bunting", "Snowy Owl", "Solitaire", "Sparrow", "Spoonbill", "Spoonbill", "Spruce Grouse",
    "Starling", "Steller's Jay", "Stilt", "Stork", "Sulphur-crested Cockatoo", "Sunbittern", "Sunbird",
    "Swallow", "Swan", "Tanager", "Tawny Owl", "Teal", "Tern", "Thick-billed Murre", "Thrasher", "Thrush",
    "Toucan", "Turkey", "Umbrellabird", "Upland Sandpiper", "Vulture", "Wagtail", "Warbler",
    "Weaver Bird", "Whippoorwill", "White Pelican", "Wigeon", "Willet", "Wood Duck", "Woodpecker", "Wren",
    "Yellow Warbler", "Yellow-bellied Sapsucker", "Yellow-crowned Night Heron", "Yellow-headed Blackbird",
    "Yellow-legged Gull", "Yellow-rumped Warbler", "Yellow-throated Warbler", "Zebra Dove", "Zebra Finch"
]

fish_list = [
    "Anchovy", "Angelfish", "Archerfish", "Arowana", "Barracuda", "Barreleye", "Bass", "Betta Fish",
    "Blue Marlin", "Bluegill", "Bonefish", "Bowfin", "Bream", "Brook Trout", "Brown Trout",
    "Butterflyfish", "Caiman", "Carp", "Catfish", "Clownfish", "Cod", "Coelacanth", "Darter",
    "Discus", "Dogfish", "Dory", "Dragonet", "Eel", "Electric Ray", "Flounder", "Flying Fish",
    "Goby", "Goldfish", "Gourami", "Grayling", "Gudgeon", "Gulper Eel", "Haddock", "Halibut",
    "Hammerhead Shark", "Herring", "Hogfish", "Icefish", "Jack Dempsey", "Jellyfish", "Koi",
    "Lamprey", "Lionfish", "Longfin Smelt", "Mackerel", "Mahi-mahi", "Marlin", "Minnow",
    "Mola Mola (Sunfish)", "Monkfish", "Moorish Idol", "Moray Eel", "Napoleon Wrasse",
    "Needlefish", "Northern Pike", "Nurse Shark", "Ocean Sunfish", "Oscar", "Parrotfish",
    "Peacock Bass", "Perch", "Permit", "Pickerel", "Piranha", "Plaice", "Pollock", "Pufferfish",
    "Queen Angelfish", "Queen Triggerfish", "Rainbow Trout", "Ratfish", "Ray", "Red Snapper",
    "Rockfish", "Salmon", "Sardine", "Sawfish", "Sculpin", "Seahorse", "Sergeant Major",
    "Sheepshead", "Shovelnose Sturgeon", "Siamese Fighting Fish (Betta)", "Silver Carp",
    "Sockeye Salmon", "Southern Flounder", "Spadefish", "Spot", "Spotted Gar", "Sprat",
    "Squirrelfish", "Sturgeon", "Surgeonfish", "Swordfish", "Tarpon", "Thresher Shark",
    "Tiger Barb", "Triggerfish", "Triplefin", "Trout", "Tuna", "Wahoo", "Walleye",
    "Warty Sea Cucumber", "Wels Catfish", "Whale Shark", "White Bass", "White Sturgeon",
    "Wobbegong Shark", "Yellow Perch", "Yellowfin Tuna", "Zebra Danio", "Zebrafish"
]



organisms_list = animal_list + bird_list + fish_list

organisms_list = list(set(organisms_list))





tokenizer = AutoTokenizer.from_pretrained(
    "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3",
    use_fast=False,
    padding_side="left",
    trust_remote_code=True,
)

generate_text = pipeline(
    model="h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3",
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    use_fast=False,
    device_map={"": "cuda:0"},
)

res = generate_text(
    "Why is drinking water so healthy?",
    min_new_tokens=2,
    max_new_tokens=1024,
    do_sample=False,
    num_beams=1,
    temperature=float(0.3),
    repetition_penalty=float(1.2),
    renormalize_logits=True
)
#print(res[0]["generated_text"])


prompts = []

def prompt_gen(token):
  description = "Short description of" + token
  res = generate_text(
    description,
    min_new_tokens=2,
    max_new_tokens=1024,
    do_sample=False,
    num_beams=1,
    temperature=float(0.3),
    repetition_penalty=float(1.2),
    renormalize_logits=True
  )
  # print(res[0]["generated_text"])
  prompts.append(res[0]["generated_text"])

key = 0

for i in range (len(organisms_list)):
  prompt_gen(organisms_list[i])
  key += 1
  print(key)
  if key == 3:
    break




