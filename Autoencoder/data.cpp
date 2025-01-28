#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

using namespace std;

int main() {
    // Path to the binary file
    const string filename = "Pf25.binLE.raw_corrected_2_subsampled";

    // Number of matrices and dimensions
    const int num_matrices = 50;
    const int rows = 250;
    const int cols = 250;

    // Total number of entries
    const int total_entries = num_matrices * rows * cols;
    vector<vector<vector<float> > > data(num_matrices, vector<vector<float> >(rows, vector<float>(cols, 0)));

    // Open the binary file
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Cannot open file!\n";
        return 1;
    }

    // Read the data
    vector<float> buffer(total_entries);
    if (file.read(reinterpret_cast<char*>(buffer.data()), total_entries * sizeof(float))) {
        // Index for buffer
        int index = 0;

        // Fill the 3D vector according to the specified pattern
        for (int k = 0; k < cols; ++k) {
            for (int j = 0; j < rows; ++j) {
                for (int i = 0; i < num_matrices; ++i) {
                    data[i][j][k] = buffer[index++];
                }
            }
        }
    } else {
        cerr << "Error reading file!\n";
        return 1;
    }

    file.close();

    // Output data to text files
    for (int i = 0; i < num_matrices; ++i) {
        ofstream output_file("Matrix_" + to_string(i) + ".txt");
        if (!output_file) {
            cerr << "Cannot open output file!\n";
            return 1;
        }

        output_file << fixed << setprecision(6);
        for (int j = 0; j < rows; ++j) {
            for (int k = 0; k < cols; ++k) {
                output_file << data[i][j][k] << " ";
            }
            output_file << "\n";
        }

        output_file.close();
        cout << "Matrix_" << i << ".txt created successfully." << endl;
    }

    return 0;
}