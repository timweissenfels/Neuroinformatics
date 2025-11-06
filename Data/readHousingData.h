//
// Created by UI703201 on 03.11.2025.
//

#ifndef READHOUSINGDATA_H
#define READHOUSINGDATA_H

#include "csv.h"

struct HousingRecord {
    float longitude;
    float latitude;
    float housingMedianAge;
    float totalRooms;
    float totalBedrooms;
    float population;
    float households;
    float medianIncome;
    float medianHouseValue;
    float oceanProximity;
};

std::vector<HousingRecord> readHousingData(const std::string& filename) {
    std::vector<HousingRecord> records;
    io::CSVReader<10> in(filename);
    in.read_header(io::ignore_extra_column,
                   "longitude", "latitude", "housing_median_age", "total_rooms",
                   "total_bedrooms", "population", "households", "median_income",
                   "median_house_value", "ocean_proximity");

    HousingRecord record;
    std::string proximityStr;

    while (in.read_row(record.longitude, record.latitude, record.housingMedianAge,
                       record.totalRooms, record.totalBedrooms, record.population,
                       record.households, record.medianIncome, record.medianHouseValue,
                       proximityStr)) {
        if (record.totalBedrooms == NULL) {
            record.totalBedrooms = 1;
        }
        if (proximityStr == "INLAND") {
            record.oceanProximity = 1;
        } else if (proximityStr == "NEAR BAY") {
            record.oceanProximity = 2;
        } else if (proximityStr == "NEAR OCEAN") {
            record.oceanProximity = 3;
        } else if (proximityStr == "<1H OCEAN") {
            record.oceanProximity = 4;
        } else if (proximityStr == "ISLAND") {
            record.oceanProximity = 5;
        } else {
            record.oceanProximity = -1;
        }

        records.push_back(record);
    }

    return records;
}

std::pair<Math::Matrix<float>, Math::Matrix<float>> getXandYVectors(const std::vector<HousingRecord> &records) {
    using namespace Math;
    Matrix<float> X(12, records.size(), 0); // 9 features
    Matrix<float> Y(1, records.size(), 0); // target variable

    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& r = records[i];
        X(0, i) = r.longitude;
        X(1, i) = r.latitude;
        X(2, i) = r.housingMedianAge;
        X(3, i) = r.totalRooms;
        X(4, i) = r.totalBedrooms;
        X(5, i) = r.population;
        X(6, i) = r.households;
        X(7,i) = (r.totalBedrooms+1)/(r.households+1); // bedrooms per household
        X(8, i) = (r.totalBedrooms+1)/(r.totalRooms+1); // bedrooms per room
        X(9, i) = (r.population+1)/(r.households+1);
        X(10, i) = r.medianIncome;
        X(11, i) = r.oceanProximity; // categorical feature as numeric (not optimal) one-hot-encoding better?

        Y(0, i) = r.medianHouseValue; // Target label
    }

    return std::make_pair(X, Y);
}

#endif //READHOUSINGDATA_H