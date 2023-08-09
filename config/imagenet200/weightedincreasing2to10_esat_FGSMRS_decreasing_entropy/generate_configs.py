import os
import torch
import numpy as np

run_tool_one = 'python -m tool.train --config'
run_tool_half = 'python -m tool.train --config'
run_tool_full = 'python -m tool.train --config'

with open('./template.yaml', 'r') as f:
    template = f.read()
name_template = 'resnet50_A{nA}.FGSMRSs3.75i1e3.0_B{nB}.none'
ranking_filep = 'exp/imagenet200/vanilla/resnet50_mpO1_fcdump/FCdump/train/instancewise_decreasing_entropy_ranking.pth'
indices = torch.load('../../../'+ranking_filep).tolist()

f_run = open('run_all.sh', 'w')
f_eval = open('eval_all.sh', 'w')

labels = ['stingray', 'goldfinch, Carduelis carduelis', 'junco, snowbird', 'robin, American robin, Turdus migratorius', 'jay', 'bald eagle, American eagle, Haliaeetus leucocephalus', 'vulture', 'eft', 'bullfrog, Rana catesbeiana', 'box turtle, box tortoise', 'common iguana, iguana, Iguana iguana', 'agama', 'African chameleon, Chamaeleo chamaeleon', 'American alligator, Alligator mississipiensis', 'garter snake, grass snake', 'harvestman, daddy longlegs, Phalangium opilio', 'scorpion', 'tarantula', 'centipede', 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita', 'lorikeet', 'hummingbird', 'toucan', 'drake', 'goose', 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus', 'jellyfish', 'sea anemone, anemone', 'flatworm, platyhelminth', 'snail', 'crayfish, crawfish, crawdad, crawdaddy', 'hermit crab', 'flamingo', 'American egret, great white heron, Egretta albus', 'oystercatcher, oyster catcher', 'pelican', 'sea lion', 'Chihuahua', 'golden retriever', 'Rottweiler', 'German shepherd, German shepherd dog, German police dog, alsatian', 'pug, pug-dog', 'red fox, Vulpes vulpes', 'Persian cat', 'lynx, catamount', 'lion, king of beasts, Panthera leo', 'American black bear, black bear, Ursus americanus, Euarctos americanus', 'mongoose', 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant, emmet, pismire', 'grasshopper, hopper', 'walking stick, walkingstick, stick insect', 'cockroach, roach', 'mantis, mantid', 'leafhopper', "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk", 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus', 'cabbage butterfly', 'lycaenid, lycaenid butterfly', 'starfish, sea star', 'wood rabbit, cottontail, cottontail rabbit', 'porcupine, hedgehog', 'fox squirrel, eastern fox squirrel, Sciurus niger', 'marmot', 'bison', 'skunk, polecat, wood pussy', 'armadillo', 'baboon', 'capuchin, ringtail, Cebus capucinus', 'African elephant, Loxodonta africana', 'puffer, pufferfish, blowfish, globefish', "academic gown, academic robe, judge's robe", 'accordion, piano accordion, squeeze box', 'acoustic guitar', 'airliner', 'ambulance', 'apron', 'balance beam, beam', 'balloon', 'banjo', 'barn', 'barrow, garden cart, lawn cart, wheelbarrow', 'basketball', 'beacon, lighthouse, beacon light, pharos', 'beaker', 'bikini, two-piece', 'bow', 'bow tie, bow-tie, bowtie', 'breastplate, aegis, egis', 'broom', 'candle, taper, wax light', 'canoe', 'castle', 'cello, violoncello', 'chain', 'chest', 'Christmas stocking', 'cowboy boot', 'cradle', 'dial telephone, dial phone', 'digital clock', 'doormat, welcome mat', 'drumstick', 'dumbbell', 'envelope', 'feather boa, boa', 'flagpole, flagstaff', 'forklift', 'fountain', 'garbage truck, dustcart', 'goblet', 'go-kart', 'golfcart, golf cart', 'grand piano, grand', 'hand blower, blow dryer, blow drier, hair dryer, hair drier', 'iron, smoothing iron', "jack-o'-lantern", 'jeep, landrover', 'kimono', 'lighter, light, igniter, ignitor', 'limousine, limo', 'manhole cover', 'maraca', 'marimba, xylophone', 'mask', 'mitten', 'mosque', 'nail', 'obelisk', 'ocarina, sweet potato', 'organ, pipe organ', 'parachute, chute', 'parking meter', 'piggy bank, penny bank', 'pool table, billiard table, snooker table', 'puck, hockey puck', 'quill, quill pen', 'racket, racquet', 'reel', 'revolver, six-gun, six-shooter', 'rocking chair, rocker', 'rugby ball', 'saltshaker, salt shaker', 'sandal', 'sax, saxophone', 'school bus', 'schooner', 'sewing machine', 'shovel', 'sleeping bag', 'snowmobile', 'snowplow, snowplough', 'soap dispenser', 'spatula', "spider web, spider's web", 'steam locomotive', 'stethoscope', 'studio couch, day bed', 'submarine, pigboat, sub, U-boat', 'sundial', 'suspension bridge', 'syringe', 'tank, army tank, armored combat vehicle, armoured combat vehicle', 'teddy, teddy bear', 'toaster', 'torch', 'tricycle, trike, velocipede', 'umbrella', 'unicycle, monocycle', 'viaduct', 'volleyball', 'washer, automatic washer, washing machine', 'water tower', 'wine bottle', 'wreck', 'guacamole', 'pretzel', 'cheeseburger', 'hotdog, hot dog, red hot', 'broccoli', 'cucumber, cuke', 'bell pepper', 'mushroom', 'lemon', 'banana', 'custard apple', 'pomegranate', 'carbonara', 'bubble', 'cliff, drop, drop-off', 'volcano', 'ballplayer, baseball player', 'rapeseed', "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum", 'corn', 'acorn']

N = 10
AuB = list(range(N))
As = AuB
ninstances = len(indices)
bin_width = ninstances // N

for i in range(1,N):
    A = As[:i]
    A = sorted(A)
    B = [j for j in range(N) if j not in A]

    nA = len(A)
    nB = N - nA

    A_factor = 1.0 + i
    weight_value_A = A_factor * ((N**2 - N*i) / (N*i))
    weight_value_B = 1.0

    print(nA, weight_value_A, weight_value_B)

    cfg = template.format(A=str(A), B=str(B), nbins=N, ranking_filep=ranking_filep, loss_indices=indices[:i*bin_width], weight_value_A=weight_value_A, weight_value_B=weight_value_B, num_instances=ninstances)
    filename = name_template.format(nA=nA, nB=nB)

    assert not os.path.exists(filename)
    with open(filename+'.yaml', 'w') as f:
        f.write(cfg)

    if nA == 1:
        f_run.write(f'{run_tool_one} config/imagenet200/weightedincreasing2to10_esat_FGSMRS_decreasing_entropy/{filename}.yaml\n')
    elif nA < 5:
        f_run.write(f'{run_tool_half} config/imagenet200/weightedincreasing2to10_esat_FGSMRS_decreasing_entropy/{filename}.yaml\n')
    else:
        f_run.write(f'{run_tool_full} config/imagenet200/weightedincreasing2to10_esat_FGSMRS_decreasing_entropy/{filename}.yaml\n')

    f_eval.write("python -m tool.train --config config/imagenet200/weightedincreasing2to10_esat_FGSMRS_decreasing_entropy/{name}.yaml \
training.resume='exp/imagenet200/weightedincreasing2to10_esat_FGSMRS_decreasing_entropy/{name}/model/best_rob_acc.pth' training.evaluate_only=True \
evaluations.auto_attack_l2.eval_freq=1 evaluations.auto_attack_l2.eval_mode=True \
evaluations.auto_attack_l2.n_samples=9000 evaluations.auto_attack_l2.batch_size=1000 \
evaluations.auto_attack_l2.save_path='exp/imagenet200/weightedincreasing2to10_esat_FGSMRS_decreasing_entropy/{name}/evaluations/auto_attack_l2' \
training.train_gpu=[0]\n".format(
            name=filename))

f_run.close()
f_eval.close()
