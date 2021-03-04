import os
root_dir = '/nas/staff/data_work/Sure/GPU_wav_dump/SE_UNet/UNet/wSDR/test_wav_trunks'
target_dir = '/nas/staff/data_work/Sure/GPU_wav_dump/SE_UNet/UNet/wSDR/test_wav'

sent_trunk = {}
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.wav'):
            elements = file.split('_')
            sent_id = elements[2]
            trunk_id = int(elements[3][:-4])
            if sent_id not in sent_trunk.keys():
                sent_trunk[sent_id] = [trunk_id]
            else:
                if trunk_id not in sent_trunk[sent_id]:
                    sent_trunk[sent_id].append(trunk_id)
                    sent_trunk[sent_id].sort()

for sent in sent_trunk.keys():
    trunks_id = sent_trunk[sent]
    for stream in ['clean', 'noisy', 'denoised']:
        source_wav = 'sox'
        for trunk in trunks_id:
            source_wav += ' {}/{}_{}_{}_{}.wav'.format(root_dir, stream, '1080000', sent, trunk)
        target_wav = '{}/{}_{}_{}.wav'.format(target_dir, stream, '1080000', sent)
        command = source_wav + ' ' + target_wav
        os.system(command)

