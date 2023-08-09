import os
import numpy as np

run_tool_one = 'python -m tool.train --config'
run_tool_half = 'python -m tool.train --config'
run_tool_full = 'python -m tool.train --config'

with open('./template.yaml', 'r') as f:
    template = f.read()
name_template = 'resnet50_A{nA}.PGDs0.5i7e3.0_B{nB}.none'


f_run = open('run_all.sh', 'w')
f_eval = open('eval_all.sh', 'w')

labels = ['stingray', 'goldfinch, Carduelis carduelis', 'junco, snowbird', 'robin, American robin, Turdus migratorius', 'jay', 'bald eagle, American eagle, Haliaeetus leucocephalus', 'vulture', 'eft', 'bullfrog, Rana catesbeiana', 'box turtle, box tortoise', 'common iguana, iguana, Iguana iguana', 'agama', 'African chameleon, Chamaeleo chamaeleon', 'American alligator, Alligator mississipiensis', 'garter snake, grass snake', 'harvestman, daddy longlegs, Phalangium opilio', 'scorpion', 'tarantula', 'centipede', 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita', 'lorikeet', 'hummingbird', 'toucan', 'drake', 'goose', 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus', 'jellyfish', 'sea anemone, anemone', 'flatworm, platyhelminth', 'snail', 'crayfish, crawfish, crawdad, crawdaddy', 'hermit crab', 'flamingo', 'American egret, great white heron, Egretta albus', 'oystercatcher, oyster catcher', 'pelican', 'sea lion', 'Chihuahua', 'golden retriever', 'Rottweiler', 'German shepherd, German shepherd dog, German police dog, alsatian', 'pug, pug-dog', 'red fox, Vulpes vulpes', 'Persian cat', 'lynx, catamount', 'lion, king of beasts, Panthera leo', 'American black bear, black bear, Ursus americanus, Euarctos americanus', 'mongoose', 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant, emmet, pismire', 'grasshopper, hopper', 'walking stick, walkingstick, stick insect', 'cockroach, roach', 'mantis, mantid', 'leafhopper', "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk", 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus', 'cabbage butterfly', 'lycaenid, lycaenid butterfly', 'starfish, sea star', 'wood rabbit, cottontail, cottontail rabbit', 'porcupine, hedgehog', 'fox squirrel, eastern fox squirrel, Sciurus niger', 'marmot', 'bison', 'skunk, polecat, wood pussy', 'armadillo', 'baboon', 'capuchin, ringtail, Cebus capucinus', 'African elephant, Loxodonta africana', 'puffer, pufferfish, blowfish, globefish', "academic gown, academic robe, judge's robe", 'accordion, piano accordion, squeeze box', 'acoustic guitar', 'airliner', 'ambulance', 'apron', 'balance beam, beam', 'balloon', 'banjo', 'barn', 'barrow, garden cart, lawn cart, wheelbarrow', 'basketball', 'beacon, lighthouse, beacon light, pharos', 'beaker', 'bikini, two-piece', 'bow', 'bow tie, bow-tie, bowtie', 'breastplate, aegis, egis', 'broom', 'candle, taper, wax light', 'canoe', 'castle', 'cello, violoncello', 'chain', 'chest', 'Christmas stocking', 'cowboy boot', 'cradle', 'dial telephone, dial phone', 'digital clock', 'doormat, welcome mat', 'drumstick', 'dumbbell', 'envelope', 'feather boa, boa', 'flagpole, flagstaff', 'forklift', 'fountain', 'garbage truck, dustcart', 'goblet', 'go-kart', 'golfcart, golf cart', 'grand piano, grand', 'hand blower, blow dryer, blow drier, hair dryer, hair drier', 'iron, smoothing iron', "jack-o'-lantern", 'jeep, landrover', 'kimono', 'lighter, light, igniter, ignitor', 'limousine, limo', 'manhole cover', 'maraca', 'marimba, xylophone', 'mask', 'mitten', 'mosque', 'nail', 'obelisk', 'ocarina, sweet potato', 'organ, pipe organ', 'parachute, chute', 'parking meter', 'piggy bank, penny bank', 'pool table, billiard table, snooker table', 'puck, hockey puck', 'quill, quill pen', 'racket, racquet', 'reel', 'revolver, six-gun, six-shooter', 'rocking chair, rocker', 'rugby ball', 'saltshaker, salt shaker', 'sandal', 'sax, saxophone', 'school bus', 'schooner', 'sewing machine', 'shovel', 'sleeping bag', 'snowmobile', 'snowplow, snowplough', 'soap dispenser', 'spatula', "spider web, spider's web", 'steam locomotive', 'stethoscope', 'studio couch, day bed', 'submarine, pigboat, sub, U-boat', 'sundial', 'suspension bridge', 'syringe', 'tank, army tank, armored combat vehicle, armoured combat vehicle', 'teddy, teddy bear', 'toaster', 'torch', 'tricycle, trike, velocipede', 'umbrella', 'unicycle, monocycle', 'viaduct', 'volleyball', 'washer, automatic washer, washing machine', 'water tower', 'wine bottle', 'wreck', 'guacamole', 'pretzel', 'cheeseburger', 'hotdog, hot dog, red hot', 'broccoli', 'cucumber, cuke', 'bell pepper', 'mushroom', 'lemon', 'banana', 'custard apple', 'pomegranate', 'carbonara', 'bubble', 'cliff, drop, drop-off', 'volcano', 'ballplayer, baseball player', 'rapeseed', "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum", 'corn', 'acorn']

As = [157, 152, 165, 106, 118, 123, 131, 126, 91, 85, 171, 90, 140, 119, 160, 156, 107, 172, 128, 142, 168, 53, 55, 108, 94, 153, 93, 147, 170, 101, 102, 92, 192, 83, 99, 146, 193, 177, 112, 30, 105, 37, 98, 80, 148, 122, 56, 76, 151, 133, 143, 169, 144, 137, 114, 161, 178, 31, 117, 88, 136, 29, 127, 163, 57, 6, 36, 110, 175, 167, 77, 47, 54, 100, 89, 198, 141, 81, 66, 13, 75, 103, 109, 164, 0, 199, 10, 111, 16, 135, 186, 182, 46, 87, 113, 184, 97, 41, 52, 188, 194, 71, 18, 116, 67, 104, 124, 12, 58, 95, 64, 11, 69, 63, 44, 40, 72, 82, 24, 49, 162, 38, 129, 121, 21, 70, 174, 50, 65, 139, 155, 84, 51, 17, 39, 43, 138, 42, 15, 180, 195, 33, 59, 74, 48, 35, 132, 5, 115, 185, 8, 96, 154, 2, 187, 158, 45, 176, 189, 86, 179, 22, 166, 26, 173, 19, 3, 79, 183, 28, 190, 68, 27, 7, 4, 145, 134, 23, 181, 130, 25, 14, 150, 73, 62, 34, 9, 61, 78, 159, 1, 120, 149, 20, 191, 125, 196, 32, 197, 60]

nbins = 10
N = len(As)
bin_width = N//nbins

for i in range(1,nbins):
    A = As[:i*bin_width]
    A = sorted(A)
    B = [j for j in range(len(labels)) if j not in A]
    B = sorted(B)

    nA = i
    nB = nbins - i

    A_factor = 1.0 + i
    weight = np.full((N,), 1.0)
    for idx in A:
        weight[idx] = A_factor * (N*N - N*nA*bin_width) / (N*nA*bin_width)
    print(nA, A_factor, weight[A[0]])

    cfg = template.format(A=str(A), B=str(B), weight=str(weight.tolist()))
    filename = name_template.format(nA=nA, nB=nB)

    assert not os.path.exists(filename)
    with open(filename+'.yaml', 'w') as f:
        f.write(cfg)

    if nA == 1:
        f_run.write(f'{run_tool_one} config/imagenet200/weightedincreasing2to10_csat_pgd7_decreasing_entropy/{filename}.yaml\n')
    elif nA < 5:
        f_run.write(f'{run_tool_half} config/imagenet200/weightedincreasing2to10_csat_pgd7_decreasing_entropy/{filename}.yaml\n')
    else:
        f_run.write(f'{run_tool_full} config/imagenet200/weightedincreasing2to10_csat_pgd7_decreasing_entropy/{filename}.yaml\n')

    f_eval.write("python -m tool.train --config config/imagenet200/weightedincreasing2to10_csat_pgd7_decreasing_entropy/{name}.yaml \
training.resume='exp/imagenet200/weightedincreasing2to10_csat_pgd7_decreasing_entropy/{name}/model/best_rob_acc.pth' training.evaluate_only=True \
evaluations.auto_attack_l2.eval_freq=1 evaluations.auto_attack_l2.eval_mode=True \
evaluations.auto_attack_l2.n_samples=9000 evaluations.auto_attack_l2.batch_size=100 \
evaluations.auto_attack_l2.save_path='exp/imagenet200/weightedincreasing2to10_csat_pgd7_decreasing_entropy/{name}/evaluations/auto_attack_l2' \
training.train_gpu=[0]\n".format(
            name=filename))

f_run.close()
f_eval.close()
