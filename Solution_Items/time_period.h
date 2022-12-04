#ifndef TIME_PERIOD_H
#define TIME_PERIOD_H

class time_period{
public:
	float t_start;
	float t_end;

	bool operator<(const float& t) {
		return (this->t_end < t);
	}

	bool operator>(const float& t) {
		return (this->t_start > t);
	}

	bool operator==(const float& t) {
		return (this->t_start <= t) && (this->t_end >= t);
	}

};

#endif