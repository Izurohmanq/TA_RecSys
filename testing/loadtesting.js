import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  // A number specifying the number of VUs to run concurrently.
  vus: 5,
  // A string specifying the total duration of the test run.
  duration: "30s",
  thresholds: {
    http_req_failed: ["rate<0.01"], // error rate should be less than 1%
  },
};

export default function () {
  const url = "http://116.193.191.147:80/nutrition";
  const payload = JSON.stringify({
    umur: 25,
    tb: 165,
    bb: 60,
    aktifitas: "bisa jalan",
    kondisi: "hamil_trim_1",
    waktu_makan: 1,
    food_names: ["apel segar"],
  });

  const params = {
    headers: {
      "Content-Type": "application/json",
    },
  };

  const res = http.post(url, payload, params);

  check(res, {
    "status was 200": (r) => r.status === 200,
    "response time < 500ms": (r) => r.timings.duration < 500,
  });

  sleep(1);
}
