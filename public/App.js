const url = require('url');

const express = require('express');
const app = express();

const path = require('path');
const fs = require('fs');
const csv = require('csv-parser');

const home = require('./home');
const researchhome = require('./researchhome');
const mainpage = require('./mainpage');
const datapage = require('./datapage');
const { spawn } = require('child_process');
const bodyParser = require('body-parser');


let markers = [];

app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

  app.get('/', (req, res) => {
    // url에서 쿼리 데이터 추출
      res.send(mainpage()); // 검색어가 없으면 그냥 홈페이지를 보여줌
  });
  
  app.get('/mainpage', (req, res) => {
    // url에서 쿼리 데이터 추출
      res.send(mainpage()); // 검색어가 없으면 그냥 홈페이지를 보여줌
  });
  app.get('/datapage', (req, res) => {
    // url에서 쿼리 데이터 추출
      res.send(datapage()); // 검색어가 없으면 그냥 홈페이지를 보여줌
  });
  
  app.get('/search', (req, res) => {
    // url에서 쿼리 데이터 추출
    const queryData = url.parse(req.url, true).query;
  
    if (queryData.gu) {
      // 쿼리 데이터에서 검색어 추출
      const searchTerm = queryData.gu.trim();
      
      // 검색어에 해당되는 csv 파일 이름 생성
      const filename = `store/store_서울특별시 ${searchTerm}.csv`;
  
      // 해당 파일이 존재하면 파일을 읽어 markers에 추가
      if (fs.existsSync(filename)) {
        const csvData = [];
        fs.createReadStream(filename)
          .pipe(csv())
          .on('data', (data) => {
            if (data.lat && data.lng) {
              csvData.push({
                content: `<div>${data.store}</div>`,
                atclFetrDesc: data.atclFetrDesc,
                flrInfo: data.flrInfo,
                atclNm: data.atclNm,
                prc: data.prc,
                direction: data.direction,
                atclNo: data.atclNo,
                tradTpNm: data.tradTpNm,
                latlng: { lat: parseFloat(data.lat), lng: parseFloat(data.lng) },
              });
            }
          })
          .on('end', () => {
            markers = csvData;
            const seoulData = require('./seoul.json');
            const searchTerm = queryData.gu.trim();
            const centerLatLng = seoulData[searchTerm];
            if (centerLatLng) {
              res.send(home(centerLatLng, markers));
            } else {
              res.send(home(null, markers));
            } // 마커 데이터를 업데이트하고 홈페이지를 보여줌
          });
      } else {
        res.send(home(null, markers)); // 해당되는 파일이 없으면 그냥 홈페이지를 보여줌
      }
    } else {
      res.send(home(null, markers)); // 검색어가 없으면 그냥 홈페이지를 보여줌
    }
  });

  app.get('/researchhome', (req, res) => {
    // url에서 쿼리 데이터 추출
    const queryData = url.parse(req.url, true).query;
  
    if (queryData.gu) {
      // 쿼리 데이터에서 검색어 추출
      const searchTerm = queryData.gu.trim();
      
      // 검색어에 해당되는 csv 파일 이름 생성
      const filename = `store/store_서울특별시 ${searchTerm}.csv`;
  
      // 해당 파일이 존재하면 파일을 읽어 markers에 추가
      if (fs.existsSync(filename)) {
        const csvData = [];
        fs.createReadStream(filename)
          .pipe(csv())
          .on('data', (data) => {
            if (data.lat && data.lng) {
              csvData.push({
                content: `<div>${data.store}</div>`,
                atclFetrDesc: data.atclFetrDesc,
                flrInfo: data.flrInfo,
                atclNm: data.atclNm,
                prc: data.prc,
                direction: data.direction,
                atclNo: data.atclNo,
                tradTpNm: data.tradTpNm,
                spc2: data.spc2,
                latlng: { lat: parseFloat(data.lat), lng: parseFloat(data.lng) },
              });
            }
          })
          .on('end', () => {
            markers = csvData;
            const seoulData = require('./seoul.json');
            const searchTerm = queryData.gu.trim();
            const centerLatLng = seoulData[searchTerm];
            if (centerLatLng) {
              res.send(researchhome(centerLatLng, markers));
            } else {
              res.send(researchhome(null, markers));
            } // 마커 데이터를 업데이트하고 홈페이지를 보여줌
          });
      } else {
        res.send(researchhome(null, markers)); // 해당되는 파일이 없으면 그냥 홈페이지를 보여줌
      }
    } else {
      res.send(researchhome(null, markers)); // 검색어가 없으면 그냥 홈페이지를 보여줌
    }
  });

  app.post('/researchhome', (req, res) => {
    // 프론트엔드에서 전달된 값을 받음
    const data = req.body;
    // 파이썬 스크립트 실행
    const pythonProcess = spawn('python', ['test.py', JSON.stringify(data)]);
  
    pythonProcess.stdout.on('data', (output) => {
      // 파이썬 스크립트로부터 출력을 받음
      const analyzedData = output.toString();
  
      // 분석 결과를 프론트엔드로 전송
      res.send(analyzedData);
    });
  });
  
  
  

  // 서버 시작
  app.listen(3000, () => {
    console.log('Server listening on port 3000');
  });
