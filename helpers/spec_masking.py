from torchaudio import transforms


class SpecMasking:
    def __init__(
        self,
        time_mask_param=8,
        freq_mask_param=5,
        p=0.2,
        iid_masks=True,
        time_masks=20,
        freq_masks=8,
    ):
        self.timeMasking = transforms.TimeMasking(
            time_mask_param=time_mask_param,
            iid_masks=iid_masks,
            p=p,
        )
        self.freqMasking = transforms.FrequencyMasking(
            freq_mask_param=freq_mask_param,
            iid_masks=iid_masks,
        )

        self.time_masks = time_masks
        self.freq_masks = freq_masks

    def compute(self, batch):
        for _ in range(self.time_masks):
            batch = self.timeMasking(batch)
        for _ in range(self.freq_masks):
            batch = self.freqMasking(batch)

        return batch
