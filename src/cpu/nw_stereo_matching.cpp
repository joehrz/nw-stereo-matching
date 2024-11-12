#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // For max()

// Scoring constants
const int MATCH_SCORE = 1;
const int MISMATCH_PENALTY = -1;
const int GAP_PENALTY = -2;

// Function to calculate max of 3 values
int max(int a, int b, int c) {
    return std::max(std::max(a, b), c);
}

// Needleman-Wunsch Algorithm
void needleman_wunsch(const std::string &seq1, const std::string &seq2) {
    int n = seq1.size();
    int m = seq2.size();

    // Create a DP matrix
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));

    // Initialize the DP matrix (gap penalties)
    for (int i = 0; i <= n; ++i) {
        dp[i][0] = i * GAP_PENALTY;
    }
    for (int j = 0; j <= m; ++j) {
        dp[0][j] = j * GAP_PENALTY;
    }

    // Fill the DP matrix
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            int match = dp[i - 1][j - 1] + (seq1[i - 1] == seq2[j - 1] ? MATCH_SCORE : MISMATCH_PENALTY);
            int delete_gap = dp[i - 1][j] + GAP_PENALTY;
            int insert_gap = dp[i][j - 1] + GAP_PENALTY;
            dp[i][j] = max(match, delete_gap, insert_gap);
        }
    }

    // Traceback to get the optimal alignment
    std::string aligned_seq1, aligned_seq2;
    int i = n, j = m;

    while (i > 0 && j > 0) {
        int score = dp[i][j];
        int score_diag = dp[i - 1][j - 1];
        int score_up = dp[i - 1][j];
        int score_left = dp[i][j - 1];

        if (score == score_diag + (seq1[i - 1] == seq2[j - 1] ? MATCH_SCORE : MISMATCH_PENALTY)) {
            aligned_seq1 = seq1[i - 1] + aligned_seq1;
            aligned_seq2 = seq2[j - 1] + aligned_seq2;
            --i;
            --j;
        } else if (score == score_up + GAP_PENALTY) {
            aligned_seq1 = seq1[i - 1] + aligned_seq1;
            aligned_seq2 = "-" + aligned_seq2;
            --i;
        } else {
            aligned_seq1 = "-" + aligned_seq1;
            aligned_seq2 = seq2[j - 1] + aligned_seq2;
            --j;
        }
    }

    // Add remaining gaps if necessary
    while (i > 0) {
        aligned_seq1 = seq1[i - 1] + aligned_seq1;
        aligned_seq2 = "-" + aligned_seq2;
        --i;
    }
    while (j > 0) {
        aligned_seq1 = "-" + aligned_seq1;
        aligned_seq2 = seq2[j - 1] + aligned_seq2;
        --j;
    }

    // Output the original sequences
    std::cout << "Original Sequences:\n";
    std::cout << "Sequence 1: " << seq1 << "\n";
    std::cout << "Sequence 2: " << seq2 << "\n\n";

    // Output the aligned sequences
    std::cout << "Optimal Alignment:\n";
    std::cout << aligned_seq1 << "\n";
    std::cout << aligned_seq2 << "\n";
}

int main() {
    std::string seq1 = "AGACTAGTTAC";
    std::string seq2 = "CGAGACGT";
    
    needleman_wunsch(seq1, seq2);

    return 0;
}



