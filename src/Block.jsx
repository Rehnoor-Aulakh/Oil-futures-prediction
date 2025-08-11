import React, { useEffect, useState } from "react";
import Chart from "./Chart";

function getFormattedDate(date) {
  const formattedDate =
    date.getUTCFullYear() +
    "-" +
    String(date.getUTCMonth() + 1).padStart(2, "0") +
    "-" +
    String(date.getUTCDate()).padStart(2, "0") +
    " " +
    String(date.getUTCHours()).padStart(2, "0") +
    ":" +
    String(date.getUTCMinutes()).padStart(2, "0") +
    " UTC";
  return formattedDate;
}

function getDate() {
  const date = new Date();

  const formattedDate = getFormattedDate(date);

  return formattedDate;
}

function getFutureDate() {
  const now = new Date();
  // Calculate the next hour (when countdown reaches 0) in UTC
  const nextHour = new Date(now);
  nextHour.setUTCHours(now.getUTCHours() + 1, 0, 0, 0); // Set to next hour at 00 minutes, 00 seconds in UTC

  const formattedDate = getFormattedDate(nextHour);
  return formattedDate;
}

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${minutes}m ${secs}s`;
}

function calculatePercentChange(currentValue, previousValue) {
  if (!previousValue || previousValue === 0) return 0;
  const percentChange = ((currentValue - previousValue) / previousValue) * 100;
  return percentChange;
}

function formatPercentChange(percentChange) {
  const sign = percentChange >= 0 ? "+" : "-";
  return `${sign}${Math.abs(percentChange).toFixed(2)}%`;
}

export default function Block() {
  const [actualPrice, setActualPrice] = useState(0);
  const [predictedPrice, setPredictedPrice] = useState(0);
  const [timeRemaining, setTimeRemaining] = useState(0);
  const [actualArray, setActualArray] = useState([]);
  const [predictedArray, setPredictedArray] = useState([]);

  //Fetch data from server every second
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5500/data");
        const data = await response.json();

        if (data.actual && data.actual.length > 0) {
          setActualPrice(data.actual[data.actual.length - 1]);
          setActualArray(data.actual);
        }
        if (data.predicted && data.predicted.length > 0) {
          setPredictedPrice(data.predicted[data.predicted.length - 1]);
          setPredictedArray(data.predicted);
        }
        if (data.timeRemaining !== undefined) {
          setTimeRemaining(data.timeRemaining);
        }
      } catch (error) {
        console.error("Error fetching data: ", error);
        setTimeRemaining(3600);
      }
    };
    fetchData();
    //set up interval to fetch data every second
    const interval = setInterval(fetchData, 1000);

    return () => clearInterval(interval);
  }, []);

  const predictedPercentChange =
    predictedArray.length >= 2
      ? calculatePercentChange(
          predictedArray[predictedArray.length - 1],
          predictedArray[predictedArray.length - 2]
        )
      : 0;

  const actualPercentChange =
    actualArray.length >= 2
      ? calculatePercentChange(
          actualArray[actualArray.length - 1],
          actualArray[actualArray.length - 2]
        )
      : 0;
  return (
    <div className="flex flex-col items-center w-full">
      {/* Price Display Block */}
      <div className="bg-cyan-600 h-64 m-16 w-4/5 rounded-lg p-5">
        <div className="text-cyan-100 text-lg">
          <h1 className="py-2">Hourly Forecast</h1>
          <p
            className={`${
              predictedPrice < actualPrice ? "text-[#ed0000]" : "text-[#00ff00]"
            }  font-extrabold my-3`}
          >
            <span className="text-3xl">${predictedPrice.toFixed(2)} </span>
            <span>{formatPercentChange(predictedPercentChange)} +1h 🔮</span>
            <br />
            <span className="font-normal text-sm">
              {getFutureDate()}, -{formatTime(timeRemaining)}
            </span>
          </p>
          <p className="font-extrabold my-3">
            <span className="text-3xl">${actualPrice.toFixed(2)} </span>
            <span>{formatPercentChange(actualPercentChange)} now</span>
            <br />
            <span className="font-normal text-sm">{getDate()}</span>
          </p>
        </div>
      </div>

      {/* Chart Block */}
      <div className="bg-gray-900 w-4/5 mx-16 mb-16 rounded-lg p-5">
        <h2 className="text-white text-xl mb-4">Price Chart</h2>
        <Chart actualArray={actualArray} predictedArray={predictedArray} />
      </div>
    </div>
  );
}
