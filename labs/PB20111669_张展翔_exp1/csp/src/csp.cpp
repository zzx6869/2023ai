#include <array>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <stack>

using namespace std;

vector<string> split(const string &str) {
    char delimiter = ',';
    vector<string> result;
    string temp;
    for (char i: str) {
        if (i == delimiter) {
            result.push_back(temp);
            temp = "";
        } else temp += i;
    }
    result.push_back(temp);
    return result;
}

class shift {
public:
    shift(const shift &other) = default;

    shift(int day, int shift_cnt) : day(day), shift_cnt(shift_cnt) {}

    // �ѵ�ǰ�Űำֵ��ָ���İ���
    bool assign(int aunt_index) {
        if (enable == -1) {
            enable = aunt_index;
            return true;
        }
        return false;
    }

    // ��ֹ��ĳ���̵İ�
    void ban(int aunt_index) {
        if (ban1 == -1) ban1 = aunt_index;
        else ban2 = aunt_index;
        banned_num++;
    }

    // �����Ƿ�ban
    bool isBanned(int aunt_index) const {
        return ban1 == aunt_index || ban2 == aunt_index;
    }

    // �Ƿ񱻸�ֵ
    bool isAssigned() const {
        return enable != -1;
    }

    int day{};            // ��ʾ�ڼ���
    int shift_cnt{};      // ��ʾ����ڼ���shift
    int enable = -1;    // ��ʾ��ǰ�ѱ�ѡ��İ��̵��±꣬-1��ʾ���shift��δ����ֵ
    int banned_num = 0; // ��ʾ��ǰshift����ֹ�İ��̵�������������С����ʽ����
    int ban1 = -1, ban2 = -1;
};

class mission {
public:
    mission(int n, int d, int s) : n(n), d(d), s(s) {
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < s; j++) {
                shifts.emplace_back(i, j);
            }
        }
        aunt_shifts = vector<int>(n, 0);
    }

    mission(const mission &other) {
        for (auto &tt: other.shifts) {
            shifts.emplace_back(tt);
        }
        n = other.n;
        d = other.d;
        s = other.s;
        aunt_shifts = vector<int>(other.aunt_shifts);
        shift_requests = vector<int>(other.shift_requests);
    }

    // �ѵõ��Ĵ�ת���ɽ������ͳ������
    vector<string> transAns(vector<vector<vector<int>>> &requests) {
        vector<string> tmp_ans;
        int cnt = 0;
        for (int i = 0; i < d; i++) {
            string ans;
            for (int j = 0; j < s; j++) {
                int arranged_aunt = shifts.at(i * s + j).enable;
                ans += to_string(arranged_aunt + 1);
                if (requests.at(i).at(j).at(arranged_aunt) == 1) {
                    cnt++;
                }
                if (j < s - 1) {
                    ans += ',';
                }
            }
            tmp_ans.push_back(ans);
        }
        tmp_ans.push_back(to_string(cnt));
        return tmp_ans;
    }

    // �жϵ�ǰ�ڵ㣬�Ƿ����е��Ű඼�ѱ���ֵ
    bool isComplete() {
        for (auto &tt: shifts) {
            if (tt.enable == -1)
                return false;
        }
        return true;
    }

    // �����Ƿ񻹿ɼ�����ֵ: ���а�û�ţ�������һ����ΪԼ���޷��Ű���
    bool CanContinue() {
        if (!isComplete()) {
            for (auto &tt: shifts) {
                if (tt.banned_num >= n)
                    return false;
            }
        }
        return true;
    }

    // ��ʼ����ǰÿ��shift�ж��ٰ��������Ű�
    void initShiftNum(vector<vector<vector<int>>> &requests) {
        for (int dd = 0; dd < d; dd++) {
            for (int ss = 0; ss < s; ss++) {
                int shift_req_num = 0;
                for (int nn = 0; nn < n; nn++) {
                    if (requests.at(dd).at(ss).at(nn)) {
                        shift_req_num++;
                    }
                }
                shift_requests.push_back(shift_req_num);
            }
        }
    }

    vector<shift> shifts;
    // ͳ����ĳ��shift�ж����˷�������
    vector<int> shift_requests;
    vector<int> aunt_shifts; // ��ǰÿ�����̱����˶��ٰ�
    int n{}, d{}, s{};
};

int main() {
    int N, D, S;
    scanf("%d,%d,%d", &N, &D, &S);
    // ��һ����ά����洢��ĳ��ĳ������Щ���������Ű�
    vector<vector<vector<int>>> requests(D, vector<vector<int>>(S, vector<int>(N, 0)));

    // ��һ����ά����洢��ĳ��ĳ������Щ���������Ű�

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            string req;
            cin >> req;
            vector<string> reqs = split(req);

            for (int s = 0; s < S; s++) {
                if (reqs[s] == "1") {
                    requests.at(j).at(s).at(i) = 1;
                }
            }
        }
    }

    stack<mission> stk;

    auto start_mission = mission(N, D, S);
    start_mission.initShiftNum(requests);
    stk.push(start_mission);

    vector<string> glob_ans;

    while (!stk.empty()) {
        auto current_mission = stk.top();
        stk.pop();

        // ��Ҫ����Ƿ��޿ɸ�ֵ
        //   ����һ�б�ʾ����ǰ���񲻿ɸ�ֵ��Լ�����������Ű಻��ȷ��Ҫ��֦�����붪������ڵ�
        if (!current_mission.CanContinue()) continue;
        //   ����һ�б�ʾ����ǰ�����Ѿ����������������֤
        if (current_mission.isComplete()) {
            bool flag = false;
            for (auto &item: current_mission.aunt_shifts) {
                if (item < D * S / N) {
                    flag = true;
                    break;
                }
            }
            if (flag) continue;
            // ͳ������Ҫ��ĸ���
            glob_ans = current_mission.transAns(requests);
            break;
        }

        // ѡ��û���Ű�İ�Σ��ٴ������İ�����ҵ�������С���Ű༯�ϵİ��
        // ���ﲻ�� : ���кô��ģ���������
        int available_aunt_num = N + 1;
        int target_index = -1;
        for (int k = 0; k < D * S; k++) {
            auto current_shift = current_mission.shifts.at(k);
            // ���㵱ǰ��η�������İ�������
            auto remains_aunt_num = current_mission.shift_requests.at(k);
            // ������δ����ֵ
            if (!current_shift.isAssigned() && (remains_aunt_num < available_aunt_num)) {
                available_aunt_num = remains_aunt_num;
                target_index = k;
            }
        }
        current_mission.shift_requests.at(target_index) = N + 1;
        // ����ѡ���ǵ� k ��(��day�죬��shift��)������Ҫ���Ǹ��� k ���ĸ����̸�ֵ
        int k_day = target_index / S;
        int k_shift = target_index % S;
        // cout << "target_index: " << target_index << ", k_shift: " << k_shift << endl;
        auto k_shift_object = current_mission.shifts.at(target_index);
        // ��Ϊ����������Ҫ�ѵ� k �����п��ܵĸ�ֵ���Ž�ȥ��Ҳ����˵������ban����������Ҫ�ţ������Ƿŵ�˳���н������Ӧ�ø�ֵ��������ջ����Ӧ�ø�ֵ�������ջ

        // ��0��λ���ǰ��̱�ţ���1��λ���ǰ��̵�ǰ�Ű�������2��λ���Ǳ�ʾ������û�������Ű�
        // ֻ��Ҫ�޳���ban�İ��̼���
        vector<tuple<int, int, int>> tmp;
        auto current_shift_status = current_mission.aunt_shifts;
        for (int i = 0; i < N; i++) {
            if (!k_shift_object.isBanned(i)) {
                tmp.emplace_back(i, current_shift_status[i], requests[k_day][k_shift][i]);
            }
        }
        // ����tmp�У��ǵڶ���Ԫ��С������ǰ�棬��һ��Ԫ���ǰ��̱�ţ�������Ԫ�����Ƿ�����
        sort(tmp.begin(), tmp.end(), [](auto &a, auto &b) {
            if (get<2>(a) != get<2>(b)) {
                return get<2>(a) < get<2>(b);
            }
            return get<1>(a) > get<1>(b);
        });
        // ������Ҫ����tmp�еĸ���T������tmp�е�˳�򴴽�T���µ�mission����
        // ��ÿ�����������´���
        //   ��ֵ��Լ����������ջ
        for (auto &t: tmp) {
            auto new_mission = mission(current_mission);
            // ��ֵ
            new_mission.shifts.at(target_index).assign(get<0>(t));
            //new_mission.assign_order.emplace_back(target_index, get<0>(t));
            new_mission.aunt_shifts.at(get<0>(t)) += 1;

            // Լ������
            if (target_index > 0) {
                new_mission.shifts.at(target_index - 1).ban(get<0>(t));
            }
            if (target_index < D * S - 1) {
                new_mission.shifts.at(target_index + 1).ban(get<0>(t));
            }
            stk.push(new_mission);
        }
    }
    // ��ӡ��
    for (auto &ss: glob_ans) {
        cout << ss << endl;
    }
    
}
