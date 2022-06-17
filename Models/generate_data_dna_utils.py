import matplotlib.pyplot as plt
import numpy as np
import itertools
import collections


class DNA():
    """
    Utility class to handle DNA sequences
    """

    SOS = "<"
    EOS = ">"
    OOS = "-"

    bases = list("ATGC")
    dinuc = [
        nuc1 + nuc2
        for nuc1, nuc2 in itertools.product(bases, bases)
    ]
    trinuc = [
        nuc1 + nuc2 + nuc3
        for nuc1, nuc2, nuc3 in itertools.product(bases, bases, bases)
    ]
    half_trinuc = [
        ['AAA', 'TTT'],
        ['AAT', 'ATT'],
        ['AAG', 'CTT'],
        ['AAC', 'GTT'],
        ['ATA', 'TAT'],
        ['ATG', 'CAT'],
        ['ATC', 'GAT'],
        ['AGA', 'TCT'],
        ['AGT', 'ACT'],
        ['AGG', 'CCT'],
        ['AGC', 'GCT'],
        ['ACA', 'TGT'],
        ['ACG', 'CGT'],
        ['ACC', 'GGT'],
        ['TAA', 'TTA'],
        ['TAG', 'CTA'],
        ['TAC', 'GTA'],
        ['TTG', 'CAA'],
        ['TTC', 'GAA'],
        ['TGA', 'TCA'],
        ['TGG', 'CCA'],
        ['TGC', 'GCA'],
        ['TCG', 'CGA'],
        ['TCC', 'GGA'],
        ['GAG', 'CTC'],
        ['GAC', 'GTC'],
        ['GTG', 'CAC'],
        ['GGG', 'CCC'],
        ['GGC', 'GCC'],
        ['GCG', 'CGC'],
        ['CAG', 'CTG'],
        ['CGG', 'CCG']
    ]
    trinuc_batch = [
        ['AAA', 'TTT', 'CCC', 'GGG'],
        ['AAC', 'GTT', 'CGC', 'GCG'],
        ['AAG', 'CTT', 'CCA', 'TGG'],
        ['AAT', 'ATT', 'CGA', 'TCG'],
        ['ACA', 'TGT', 'GCC', 'GGC'],
        ['ACC', 'GGT', 'CTC', 'GAG'],
        ['ACG', 'CGT', 'GCA', 'TGC'],
        ['ACT', 'AGT', 'TAA', 'TTA'],
        ['AGA', 'TCT', 'GTA', 'TAC'],
        ['AGC', 'GCT', 'TCA', 'TGA'],
        ['AGG', 'CCT', 'GAC', 'GTC'],
        ['ATA', 'TAT', 'CAG', 'CTG'],
        ['ATC', 'GAT', 'CTA', 'TAG'],
        ['ATG', 'CAT', 'CCG', 'CGG'],
        ['CAC', 'GTG', 'GAA', 'TTC'],
        ['CAA', 'TTG', 'GGA', 'TCC']
    ]
    tetramer_batch = [
        ['TTTA', 'TAAA', 'TTAA', 'TTAA'],
        ['TTAT', 'ATAA', 'ATAT', 'ATAT'],
        ['TATT', 'AATA', 'TATA', 'TATA'],
        ['GGGC', 'GCCC', 'GGCC', 'GGCC'],
        ['GGCG', 'CGCC', 'CGCG', 'CGCG'],
        ['GCGG', 'CCGC', 'GCGC', 'GCGC'],
        ['CGGG', 'CCCG', 'CCGG', 'CCGG'],
        ['ATTT', 'AAAT', 'AATT', 'AATT'],
        ['TTTT', 'AAAA', 'TAAT', 'ATTA'],
        ['GGGG', 'CCCC', 'GCCG', 'CGGC'],
        ['TTGA', 'TCAA', 'TTCA', 'TGAA'],
        ['TTAG', 'CTAA', 'TTAC', 'GTAA'],
        ['TGTA', 'TACA', 'TCTA', 'TAGA'],
        ['TGCG', 'CGCA', 'TGCC', 'GGCA'],
        ['TCGG', 'CCGA', 'TCGC', 'GCGA'],
        ['TATG', 'CATA', 'TATC', 'GATA'],
        ['GTGC', 'GCAC', 'GTCC', 'GGAC'],
        ['GTAT', 'ATAC', 'CTAT', 'ATAG'],
        ['GGTC', 'GACC', 'GCTC', 'GAGC'],
        ['GGCT', 'AGCC', 'CGCT', 'AGCG'],
        ['GCGT', 'ACGC', 'CCGT', 'ACGG'],
        ['GATT', 'AATC', 'CATT', 'AATG'],
        ['CTGG', 'CCAG', 'CTCG', 'CGAG'],
        ['CGTG', 'CACG', 'CCTG', 'CAGG'],
        ['ATGT', 'ACAT', 'ATCT', 'AGAT'],
        ['AGTT', 'AACT', 'ACTT', 'AAGT'],
        ['TTTG', 'CAAA', 'TAAG', 'CTTA'],
        ['TTTC', 'GAAA', 'TAAC', 'GTTA'],
        ['TTGT', 'ACAA', 'TCAT', 'ATGA'],
        ['TTCT', 'AGAA', 'TGAT', 'ATCA'],
        ['TGTT', 'AACA', 'TACT', 'AGTA'],
        ['TGGG', 'CCCA', 'TCCG', 'CGGA'],
        ['TGGC', 'GCCA', 'TCCC', 'GGGA'],
        ['TGGA', 'TCCA', 'TGCA', 'TGCA'],
        ['TGCT', 'AGCA', 'AGCT', 'AGCT'],
        ['TCTT', 'AAGA', 'TAGT', 'ACTA'],
        ['TCGT', 'ACGA', 'TCGA', 'TCGA'],
        ['GTTT', 'AAAC', 'GAAT', 'ATTC'],
        ['GTTC', 'GAAC', 'GTAC', 'GTAC'],
        ['GTGG', 'CCAC', 'GCAG', 'CTGC'],
        ['GTCG', 'CGAC', 'GGAG', 'CTCC'],
        ['GTAG', 'CTAC', 'CTAG', 'CTAG'],
        ['GGTG', 'CACC', 'GACG', 'CGTC'],
        ['GGGT', 'ACCC', 'GCCT', 'AGGC'],
        ['GCTG', 'CAGC', 'GAGG', 'CCTC'],
        ['GATG', 'CATC', 'GATC', 'GATC'],
        ['CTTT', 'AAAG', 'CAAT', 'ATTG'],
        ['CTTG', 'CAAG', 'CATG', 'CATG'],
        ['CGGT', 'ACCG', 'CCCT', 'AGGG'],
        ['AGGT', 'ACCT', 'ACGT', 'ACGT'],
        ['TTGG', 'CCAA', 'TCAG', 'CTGA'],
        ['TTGC', 'GCAA', 'TCAC', 'GTGA'],
        ['TTCG', 'CGAA', 'TGAG', 'CTCA'],
        ['TTCC', 'GGAA', 'TGAC', 'GTCA'],
        ['TGTG', 'CACA', 'TACG', 'CGTA'],
        ['TGTC', 'GACA', 'TACC', 'GGTA'],
        ['TGGT', 'ACCA', 'TCCT', 'AGGA'],
        ['TCTG', 'CAGA', 'TAGG', 'CCTA'],
        ['TCTC', 'GAGA', 'TAGC', 'GCTA'],
        ['GTTG', 'CAAC', 'GAAG', 'CTTC'],
        ['GTGT', 'ACAC', 'GCAT', 'ATGC'],
        ['GTCT', 'AGAC', 'GGAT', 'ATCC'],
        ['GGTT', 'AACC', 'GACT', 'AGTC'],
        ['GCTT', 'AAGC', 'GAGT', 'ACTC'],
        ['CTGT', 'ACAG', 'CCAT', 'ATGG'],
        ['CTCT', 'AGAG', 'CGAT', 'ATCG'],
        ['CGTT', 'AACG', 'CACT', 'AGTG'],
        ['CCTT', 'AAGG', 'CAGT', 'ACTG']
    ]
    tokens = [SOS] + bases + [EOS]
    bases_oos = [OOS] + bases
    token2reverse = {
        SOS: EOS, "A": "T", "T": "A", "G": "C", "C": "G", EOS: SOS
    }
    n_bases = len(bases)
    n_bases_oos = len(bases_oos)
    n_tokens = len(tokens)

    def __init__(self):
        pass

    def generate(self, size, with_limits=False):
        seq = np.random.choice(self.bases, size=size)
        seq = "".join(seq)
        if with_limits:
            seq = self.SOS + seq + self.EOS
        return seq

    def reverse(self, seq):
        r_seq = [self.token2reverse[token] for token in reversed(seq)]
        r_seq = "".join(r_seq)
        return r_seq

    def most_common(self, seq, oligo_size, n_top, freq=False):
        seq_oligos = [seq[i:i+oligo_size] for i in range(len(seq))]
        top_oligos = collections.Counter(seq_oligos).most_common(n_top)
        if freq:
            return top_oligos
        else:
            return [nuc for nuc, freq in top_oligos]

    def seq2array(self, seq, tokens):
        token2index = {
            token: index for index, token in enumerate(tokens)
        }
        array = np.array([token2index[token] for token in seq])
        return array

    def array2seq(self, array, tokens):
        index2token = {
            index: token for index, token in enumerate(tokens)
        }
        seq = "".join([index2token[index] for index in array])
        return seq

    def indicator2seq(self, indicator, tokens):
        indexes = indicator.argmax(axis=-1)
        return self.array2seq(indexes, tokens)

    def check_oligos(self, oligos):
        all_str = all(isinstance(oligo, str) for oligo in oligos)
        all_list = all(isinstance(oligo, list) for oligo in oligos)
        if not (all_str or all_list):
            raise ValueError(
                "oligos must be a list of str (no batch) "
                "or a list of list of str (batch)"
            )
        if all_str:
            oligo_size = len(oligos[0])
            all_same_size = all(len(oligo) == oligo_size for oligo in oligos)
            oligo_to_idx = {
                oligo: idx for idx, oligo in enumerate(oligos)
            }
        if all_list:
            oligo_size = len(oligos[0][0])
            all_same_size = all(
                len(oligo) == oligo_size for batch in oligos for oligo in batch
            )
            oligo_to_idx = {
                oligo: idx
                for idx, batch in enumerate(oligos) for oligo in batch
            }
        if not all_same_size:
            raise ValueError("all oligos should have the same size")
        return oligo_size, oligo_to_idx

    def iterate(self, seq, oligos):
        """Iterate along seq.

        Yields
        ------
        - tuple x, oligo_idx
        - x: position of the oligo in the sequence
        - oligo_idx: index of the oligo in the oligo list
        """
        oligo_size, oligo_to_idx = self.check_oligos(oligos)
        for x, _ in enumerate(seq):
            oligo = seq[x: x + oligo_size]
            if oligo in oligo_to_idx:
                yield x, oligo_to_idx[oligo]

    def indicator(self, seq, oligos):
        """Return the len(seq)*len(oligos) indicator matrix.
        """
        M = np.zeros((len(seq), len(oligos)))
        for x, oligo_idx in self.iterate(seq, oligos):
            M[x, oligo_idx] = 1.
        return M

    def plot_indicator(self, seq, oligos):
        """Plot the len(seq)*len(oligos) indicator matrix.
        """
        M = self.indicator(seq, oligos)
        fig, ax = plt.subplots(1, 1, figsize=(12, 3))
        ax.imshow(M, origin="lower", aspect="auto")
        ax.RL_set(title="indicator", ylabel="sequence position")
        ax.set_xticks(range(len(oligos)))
        ax.set_xticklabels(oligos, rotation=90)
        fig.tight_layout()


def levenshtein(s, t):
    """Edit distance between strings s and t.

    References
    - https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance
    """
    if s == t:
        return 0
    elif len(s) == 0:
        return len(t)
    elif len(t) == 0:
        return len(s)
    v0 = [None] * (len(t) + 1)
    v1 = [None] * (len(t) + 1)
    for i in range(len(v0)):
        v0[i] = i
    for i in range(len(s)):
        v1[0] = i + 1
        for j in range(len(t)):
            cost = 0 if s[i] == t[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j in range(len(v0)):
            v0[j] = v1[j]

    return v1[len(t)]
