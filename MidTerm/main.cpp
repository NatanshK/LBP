#include <bits/stdc++.h>
using namespace std;
mt19937 rng(42);

struct Portfolio {
    vector<double> weights;
    vector<int> selected_assets;

    double expected_return;
    double expected_shortfall;

    int rank;                 // the tier of the portfolio in the Pareto front
    double crowding_distance; // isolation from neighbours
};

vector<vector<double>> load_market_data(const string &filename) {
    vector<vector<double>> data;
    ifstream file(filename);
    string line;
    getline(file, line);

    while (getline(file, line)) {
        vector<double> row;
        stringstream ss(line);
        string cell;
        getline(ss, cell, ',');

        while (getline(ss, cell, ',')) {
            try {
                row.push_back(stod(cell));
            }
            catch (const invalid_argument &e) {
                row.push_back(0.0);
            }
        }
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    file.close();
    return data;
}

double calculate_expected_return(const Portfolio &p, const vector<double> &mean_returns) {
    double exp_ret = 0.0;
    for (int i = 0; i < p.selected_assets.size(); ++i) {
        int asset_idx = p.selected_assets[i];
        exp_ret += p.weights[i] * mean_returns[asset_idx];
    }
    return exp_ret;
}

double calculate_expected_shortfall(const Portfolio &p, const vector<vector<double>> &market_data, double alpha = 0.05) {
    int num_days = market_data.size();
    vector<double> daily_portfolio_returns(num_days, 0.0);

    for (int t = 0; t < num_days; ++t) {
        double daily_ret = 0.0;
        for (int i = 0; i < p.selected_assets.size(); ++i) {
            int asset_idx = p.selected_assets[i];
            daily_ret += p.weights[i] * market_data[t][asset_idx];
        }
        daily_portfolio_returns[t] = daily_ret;
    }

    sort(daily_portfolio_returns.begin(), daily_portfolio_returns.end());

    int k = floor(alpha * num_days);
    if (k == 0) {
        k = 1;
    }

    double sum_tail = 0.0;
    for (int t = 0; t < k; ++t) {
        sum_tail += daily_portfolio_returns[t];
    }

    return -(sum_tail / k);
}

vector<double> calculate_mean_returns(const vector<vector<double>> &market_data) {
    int num_days = market_data.size();
    int num_assets = market_data[0].size();
    vector<double> means(num_assets, 0.0);

    for (int i = 0; i < num_assets; ++i) {
        double sum = 0.0;
        for (int t = 0; t < num_days; ++t) {
            sum += market_data[t][i];
        }
        means[i] = sum / num_days;
    }
    return means;
}



vector<Portfolio> initialize_population(int pop_size, int total_assets, int K, const vector<double> &mean_returns, const vector<vector<double>> &market_data) {
    vector<Portfolio> population(pop_size);
    uniform_real_distribution<double> dist(0.01, 1.0);

    for (int p = 0; p < pop_size; ++p) {

        vector<int> all_assets(total_assets);
        iota(all_assets.begin(), all_assets.end(), 0);
        shuffle(all_assets.begin(), all_assets.end(), rng);
        population[p].selected_assets.assign(all_assets.begin(), all_assets.begin() + K);

        double sum_w = 0.0;
        for (int i = 0; i < K; ++i) {
            double w = dist(rng);
            population[p].weights.push_back(w);
            sum_w += w;
        }
        for (int i = 0; i < K; ++i) {
            population[p].weights[i] /= sum_w;
        }

        population[p].expected_return = calculate_expected_return(population[p], mean_returns);
        population[p].expected_shortfall = calculate_expected_shortfall(population[p], market_data);
    }
    return population;
}

bool dominates(const Portfolio &a, const Portfolio &b) {
    bool return_better_or_equal = a.expected_return >= b.expected_return;
    bool risk_better_or_equal = a.expected_shortfall <= b.expected_shortfall;

    bool return_strictly_better = a.expected_return > b.expected_return;
    bool risk_strictly_better = a.expected_shortfall < b.expected_shortfall;

    return (return_better_or_equal && risk_better_or_equal && (return_strictly_better || risk_strictly_better));
}

vector<vector<int>> fast_non_dominated_sort(vector<Portfolio> &pop) {
    int n = pop.size();
    vector<vector<int>> fronts;
    vector<int> front1;

    vector<int> domination_count(n, 0);
    vector<vector<int>> dominated_solutions(n);

    for (int p = 0; p < n; ++p) {
        for (int q = 0; q < n; ++q) {
            if (p == q) {
                continue;
            }
            if (dominates(pop[p], pop[q])) {
                dominated_solutions[p].push_back(q); // p dominates q
            }
            else if (dominates(pop[q], pop[p])) {
                domination_count[p]++; // q dominates p
            }
        }

        if (domination_count[p] == 0) {
            pop[p].rank = 1;
            front1.push_back(p);
        }
    }

    fronts.push_back(front1);

    int i = 0;
    while (i < fronts.size() && !fronts[i].empty()) {
        vector<int> next_front;
        for (int p : fronts[i]) {
            for (int q : dominated_solutions[p]) {
                domination_count[q]--; // Remove p's domination over q
                if (domination_count[q] == 0) {
                    pop[q].rank = i + 2; // Ranks are 1-indexed
                    next_front.push_back(q);
                }
            }
        }
        if (!next_front.empty()) {
            fronts.push_back(next_front);
        }
        i++;
    }

    return fronts;
}

void calculate_crowding_distance(vector<int> &front_indices, vector<Portfolio> &pop) {
    int l = front_indices.size();
    if (l == 0) {
        return;
    }
    for (int idx : front_indices) {
        pop[idx].crowding_distance = 0.0;
    }

    if (l <= 2) {
        for (int idx : front_indices) {
            pop[idx].crowding_distance = 1e9;
        }
        return;
    }

    sort(front_indices.begin(), front_indices.end(), [&pop](int a, int b) {
        return pop[a].expected_return < pop[b].expected_return;
    });

    pop[front_indices[0]].crowding_distance = 1e9;
    pop[front_indices[l - 1]].crowding_distance = 1e9;

    double min_ret = pop[front_indices[0]].expected_return;
    double max_ret = pop[front_indices[l - 1]].expected_return;

    if (max_ret - min_ret > 1e-9) {
        for (int i = 1; i < l - 1; ++i) {
            pop[front_indices[i]].crowding_distance +=
                (pop[front_indices[i + 1]].expected_return - pop[front_indices[i - 1]].expected_return) / (max_ret - min_ret);
        }
    }

    sort(front_indices.begin(), front_indices.end(), [&pop](int a, int b) {
        return pop[a].expected_shortfall < pop[b].expected_shortfall;
    });

    pop[front_indices[0]].crowding_distance = 1e9;
    pop[front_indices[l - 1]].crowding_distance = 1e9;

    double min_risk = pop[front_indices[0]].expected_shortfall;
    double max_risk = pop[front_indices[l - 1]].expected_shortfall;

    if (max_risk - min_risk > 1e-9) {
        for (int i = 1; i < l - 1; ++i) {
            pop[front_indices[i]].crowding_distance +=
                (pop[front_indices[i + 1]].expected_shortfall - pop[front_indices[i - 1]].expected_shortfall) / (max_risk - min_risk);
        }
    }
}

int tournament_selection(const vector<Portfolio> &pop) {

    uniform_int_distribution<int> dist(0, pop.size() - 1);

    int parent1_idx = dist(rng);
    int parent2_idx = dist(rng);

    const Portfolio &p1 = pop[parent1_idx];
    const Portfolio &p2 = pop[parent2_idx];

    if (p1.rank < p2.rank) {
        return parent1_idx;
    }
    else if (p2.rank < p1.rank) {
        return parent2_idx;
    }

    else {
        if (p1.crowding_distance > p2.crowding_distance) {
            return parent1_idx;
        }
        else {
            return parent2_idx;
        }
    }
}

Portfolio crossover(const Portfolio &p1, const Portfolio &p2, int K) {
    Portfolio child;
    vector<int> pool_assets;
    vector<double> pool_weights;

    for (int i = 0; i < K; ++i) {
        pool_assets.push_back(p1.selected_assets[i]);
        pool_weights.push_back(p1.weights[i]);
    }

    for (int i = 0; i < K; ++i) {
        auto it = find(pool_assets.begin(), pool_assets.end(), p2.selected_assets[i]);
        if (it != pool_assets.end()) {

            int idx = distance(pool_assets.begin(), it);
            pool_weights[idx] = (pool_weights[idx] + p2.weights[i]) / 2.0;
        }
        else {

            pool_assets.push_back(p2.selected_assets[i]);
            pool_weights.push_back(p2.weights[i]);
        }
    }

    vector<int> indices(pool_assets.size());
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), rng);

    double sum_w = 0.0;
    for (int i = 0; i < K; ++i) {
        int random_idx = indices[i];
        child.selected_assets.push_back(pool_assets[random_idx]);
        child.weights.push_back(pool_weights[random_idx]);
        sum_w += pool_weights[random_idx];
    }

    for (int i = 0; i < K; ++i) {
        child.weights[i] /= sum_w;
    }

    return child;
}

void mutate(Portfolio &p, int K, int total_assets, double mutation_rate = 0.1) {
    uniform_real_distribution<double> prob(0.0, 1.0);
    uniform_real_distribution<double> weight_shift(0.8, 1.2); // +/- 20% shift
    uniform_int_distribution<int> asset_dist(0, total_assets - 1);

    bool needs_repair = false;

    if (prob(rng) < mutation_rate) {
        uniform_int_distribution<int> drop_idx_dist(0, K - 1);
        int drop_idx = drop_idx_dist(rng);

        int new_asset;
        bool is_unique;
        do {
            new_asset = asset_dist(rng);
            is_unique = true;
            for (int a : p.selected_assets)
            {
                if (a == new_asset)
                    is_unique = false;
            }
        } while (!is_unique);

        p.selected_assets[drop_idx] = new_asset;
        p.weights[drop_idx] = prob(rng);
        needs_repair = true;
    }

    for (int i = 0; i < K; ++i) {
        if (prob(rng) < mutation_rate) {
            p.weights[i] *= weight_shift(rng);
            needs_repair = true;
        }
    }

    if (needs_repair) {
        double sum_w = 0.0;
        for (double w : p.weights) {
            sum_w += w;
        }
        for (int i = 0; i < K; ++i) {
            p.weights[i] /= sum_w;
        }
    }
}

vector<string> load_tickers(const string &filename) {
    vector<string> tickers;
    ifstream file(filename);
    string line, cell;

    if (file.is_open() && getline(file, line)) {
        stringstream ss(line);
        getline(ss, cell, ',');

        while (getline(ss, cell, ',')) {
            cell.erase(remove(cell.begin(), cell.end(), '\r'), cell.end());
            cell.erase(remove(cell.begin(), cell.end(), '\n'), cell.end());
            tickers.push_back(cell);
        }
    }
    return tickers;
}

int main() {
    vector<vector<double>> market_data = load_market_data("asset_log_returns.csv");
    if (market_data.empty()) {
        return 1;
    }

    int total_assets = market_data[0].size();
    int total_days = market_data.size();
    cout << "Dataset loaded. Days: " << total_days << ", Assets: " << total_assets << endl;

    vector<double> mean_returns = calculate_mean_returns(market_data);

    int pop_size = 100;
    int max_generations = 100;
    int K = 10;

    cout << "Spawning Generation 0..." << endl;
    vector<Portfolio> population = initialize_population(pop_size, total_assets, K, mean_returns, market_data);

    vector<vector<int>> initial_fronts = fast_non_dominated_sort(population);
    for (auto &front : initial_fronts) {
        calculate_crowding_distance(front, population);
    }

    for (int gen = 0; gen < max_generations; ++gen) {
        cout << "Running Generation " << gen + 1 << " / " << max_generations << "...\r" << flush;

        vector<Portfolio> offspring;
        for (int i = 0; i < pop_size; ++i) {
            int p1_idx = tournament_selection(population);
            int p2_idx = tournament_selection(population);

            Portfolio child = crossover(population[p1_idx], population[p2_idx], K);
            mutate(child, K, total_assets, 0.1);

            child.expected_return = calculate_expected_return(child, mean_returns);
            child.expected_shortfall = calculate_expected_shortfall(child, market_data);

            offspring.push_back(child);
        }

        vector<Portfolio> combined_pop = population;
        combined_pop.insert(combined_pop.end(), offspring.begin(), offspring.end());

        vector<vector<int>> fronts = fast_non_dominated_sort(combined_pop);

        vector<Portfolio> next_gen;
        int current_front = 0;

        while (next_gen.size() + fronts[current_front].size() <= pop_size) {
            calculate_crowding_distance(fronts[current_front], combined_pop);
            for (int idx : fronts[current_front]) {
                next_gen.push_back(combined_pop[idx]);
            }
            current_front++;
        }

        if (next_gen.size() < pop_size) {
            calculate_crowding_distance(fronts[current_front], combined_pop);
            vector<int> last_front = fronts[current_front];

            sort(last_front.begin(), last_front.end(), [&combined_pop](int a, int b) {
                return combined_pop[a].crowding_distance > combined_pop[b].crowding_distance;
            });

            int needed = pop_size - next_gen.size();
            for (int i = 0; i < needed; ++i) {
                next_gen.push_back(combined_pop[last_front[i]]);
            }
        }

        population = next_gen;

        vector<vector<int>> new_fronts = fast_non_dominated_sort(population);
        for (auto &front : new_fronts) {
            calculate_crowding_distance(front, population);
        }
    }

    cout << "\nEvolution Complete!" << endl;
    vector<vector<int>> final_fronts = fast_non_dominated_sort(population);
    cout << "Elite Portfolios in Final Pareto Front: " << final_fronts[0].size() << endl;

    cout << "\nSample of Top Elite Portfolios" << endl;
    for (int i = 0; i < min(5, (int)final_fronts[0].size()); ++i) {
        Portfolio &p = population[final_fronts[0][i]];
        cout << "Portfolio " << i + 1 << " -> Return: " << p.expected_return << " | Shortfall (Risk): " << p.expected_shortfall << endl;
    }

    ofstream out_file("ga_pareto_front.csv");
    out_file << "Return,Shortfall\n";
    for (int idx : final_fronts[0]) {
        out_file << population[idx].expected_return << "," << population[idx].expected_shortfall << "\n";
    }
    out_file.close();
    cout << "\nResults successfully exported to 'ga_pareto_front.csv'!\n" << endl;

    vector<string> tickers = load_tickers("asset_log_returns.csv");

    sort(final_fronts[0].begin(), final_fronts[0].end(), [&population](int a, int b) {
        return population[a].expected_shortfall < population[b].expected_shortfall;
    });

    Portfolio &safest = population[final_fronts[0].front()];
    Portfolio &aggressive = population[final_fronts[0].back()];

    cout << "THE SAFEST PORTFOLIO (Left Edge)" << endl;
    cout << "Expected Daily Return: " << safest.expected_return << endl;
    cout << "Expected Shortfall (Risk): " << safest.expected_shortfall << "\n";
    for (int i = 0; i < safest.selected_assets.size(); ++i) {
        cout << tickers[safest.selected_assets[i]] << ": " << safest.weights[i] * 100.0 << "%" << endl;
    }

    cout << "\nTHE AGGRESSIVE PORTFOLIO (Right Edge)" << endl;
    cout << "Expected Daily Return: " << aggressive.expected_return << endl;
    cout << "Expected Shortfall (Risk): " << aggressive.expected_shortfall << "\n";
    for (int i = 0; i < aggressive.selected_assets.size(); ++i) {
        cout << tickers[aggressive.selected_assets[i]] << ": "
             << aggressive.weights[i] * 100.0 << "%" << endl;
    }

    return 0;
}