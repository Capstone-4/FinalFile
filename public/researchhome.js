function researchhome(centerLatLng, markers) {
  if (!centerLatLng || !centerLatLng.lat) {
    // centerLatLng 객체가 유효하지 않거나 lat 속성이 없는 경우 기본값 설정
    centerLatLng = { lat: 37.5665, lng: 126.9780 };
  }
  return `<!DOCTYPE html>
  <html>
    <head>
      <meta charset="utf-8"/>
      <title>Kakao 지도 시작하기</title>
      <script type="text/javascript" src="//dapi.kakao.com/v2/maps/sdk.js?appkey=58ade57e1d4d5f4677a833c1b7a69515&libraries=clusterer,drawing"></script>
      <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
      <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>

      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
      <style>

      body {
            font-family: 'Maplestory';
            src: url('/Maplestory.ttf');
        }
    
        .container-fluid {
          width: 100vw;
          height: 100vh;
          position: relative;
          z-index: 10;
      }
      
      a{
          text-decoration-line: none;
      }
      
      .title {
          width: 20vw;
          height: 20vh;
          color: white;
          float: left;
          font-size: 20px;
          margin-top: 2%;
          margin-left: 5%;
      }
      .top-banner {
          width: 101.5%;
          height: 14%;
          background-color: #088bf7;
      }
      
      .collapse {
          width: 100%;
          height: 30vh;
          margin-left: -10%;
          margin-top: -4%;
      }
      .nav-link {
          margin-left: 10vw;
          margin-bottom: 11%;
          color: white;
          text-align: center;
          font-size: 20px;
      }
      .nav-link1{
        color: white;
          text-align: center;
          font-size: 20px;
      }
      
      .input-p{
          width: 20%;
          height: 75%;
          margin-left: 10px;
      }
      
      .input-title-banner{
          width: 100%;
          height: 130%;
          border-radius: 9px;
          background-color: white;
      }
      
      .input-title{
          color: #088bf7;
          margin-left:10%;
          font-size: 18px;
          margin-top: 5%;
          font-weight: 900;
      }
      
      .map {
          margin-top: 6%;
          width: 100%;
          height: 88%;
          background-color: antiquewhite;
          position: absolute;
          z-index: -1;
      }
      
      .form-group{
          margin-top: 15%;
          width: 70%;
          text-align: center;

      }
      
      .form-control {
          box-sizing: content-box;
          margin-left: 15%;
          border-radius: 7px;
          border-width: 3px;
      }
      
      .button-banner{
          text-align: center;
          margin-top: 20px;
      
      }
      
      .search-banner {
          width: 100%;
          height: 30%;
          margin-top: 10px;
      }
      
      
      h5 {
          margin-top: 5%;
          margin-left: 7%;
      }
      
      
      .three-btn-banner {
          width: 100%;
          height: 20%;
          text-align: center;
      }
      
      .three-btn {
          margin-top: 5%;
          width: 20%;
          height: 100%;
          float: left;
          margin-left: 19%;
      }
      
      .btn {
          width: 120%;
          height: 120%;
          float: left;
          font-size: 10px;
          font-weight: 1000;
          border-color: #088bf7;
          border-radius: 10px;
          border-width: 3px;
      }
      
      .ad {
          width: 30%;
          margin-left: 35%;
          margin-top: 5%;
          font-size: 12px;
      }
      
      .result-box {
          width: 20vw;
          height: 37vh;
          margin-left: 10px;
          margin-top: -24%;
          background-color: white;
          border-radius: 9px;
          display: none;
          position: absolute;
          z-index: 100;
      }

      .vl-banner {
        text-align: center;
        }
      
      .vl {
          color: black;
          margin-left: 5%;
          margin-right: 5%;
          font-size: 14px;
      }
      
      #li {
          margin-left: 10%;
          margin-top: 3%;
      }
      
      .report {
          float: right;
          width: 45%;
          height: 105%;
          background-color: white;
          margin-top: -54%;
          display: none;
          z-index: -100;
      }
      
      .report-title-banner {
          width: 100%;
          background-color: #71b2e8;
          height: 10%;
          padding-top: 4%;
          padding-left: 2%;
      }
      
      .work-banner {
          width: 20vw;
          height: 40vh;
          margin-left: 1.5%;
          border-radius: 9px;
          margin-top: -30%;
          background-color: white;
          display: none;
      }
      
      #okay2{
          margin-top: 70%;
      }
      
      .final-btn {
          margin-top: 15%;
          width: 100%;
          height: 100%;
          font-size: 15px;
      }
      
      .report2 {
          float: right;
          width: 45%;
          height: 85vh;
          background-color: white;
          margin-top: -41.4%;
          display: none;
      }
      
      .contents {
          margin-top: 5%;
          overflow-y: scroll; 
          height: 78%;
      }
      
      .pho {
          width: 50%;
          height: 50%;
          background-color: violet;
      }
      .generateJsonBtn {
          margin-left: 80px;
          margin-top: -24%;
          background-color: red;
      }
      </style>
    </head>
    <body>
    <div class="container-fluid">
    <div class="map">
    <div id="map" style="width: 100%; height: 100%;"></div>
    </div>

    <div class="title">
        <h4>empty house</h4>
        <a class="nav-link1" href="/mainpage">우리집 지금 비었는데</a>
    </div>
    <div class="top-banner">
        <div class="banner">
            <nav class="navbar navbar-expand-lg">
                <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNavDropdown" aria-controls="navbarNavDropdown" aria-expanded="false" aria-label="Toggle navigation">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNavDropdown">
                    <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link" href="/researchhome">자리 찾기</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/datapage">데이터 조회</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#">이용안내</a>
                    </li>
                    </ul>
                </div>
            </nav>
            </div>
        </div>
        <div class="input-p">
            <div class="search-banner">
                <div class="input-title-banner">
                    <h4 class="input-title" style="margin-top: 50px;">자리 찾기</h4>
                    <div class="three-btn-banner">
                        <div class="three-btn">
                            <button type="button" class="btn btn-light" id="loc-building">건물위치</button>
                        </div>
                        <div class="three-btn">
                            <button type="button" class="btn btn-light" id="anl-range">분석영역</button>
                        </div>
                    </div>
                    <form>
                        <div class="form-group col-md-4">
                            <select id="inputState" class="form-control" name="gu" onchange="createDivBox()">
                            <option selected>주소 구 단위</option>
                            <option>강남구</option>
                            <option>강동구</option>
                            <option>강북구</option>
                            <option>강서구</option>
                            <option>관악구</option>
                            <option>광진구</option>
                            <option>구로구</option>
                            <option>금천구</option>
                            <option>노원구</option>
                            <option>도봉구</option>
                            <option>동대문구</option>
                            <option>동작구</option>
                            <option>마포구</option>
                            <option>서대문구</option>
                            <option>서초구</option>
                            <option>성동구</option>
                            <option>성북구</option>
                            <option>송파구</option>
                            <option>양천구</option>
                            <option>영등포구</option>
                            <option>용산구</option>
                            <option>은평구</option>
                            <option>종로구</option>
                            <option>중구</option>
                            <option>중랑구</option>
                            </select>
                            </div>
                            <button type="submit" class="final-btn btn btn-primary" id="okay1">주소지 검색</button>
                    </form>
                    <div>
                    <button class="final-btn btn btn-primary" onclick="selectOverlay('CIRCLE')">분석 영역 설정</button>        
                    <button id="final" class="final-btn btn btn-primary" onclick="showCoordinatesAsJson()">분석 하기</button>
                    </div>
                </div>
            </div>
        </div>

        <div class="report2"> <!--주소 구 단위에서 지역을 조회하면 5개 리스트와 동시에 그 구에 있는 건물 정보 리포트 제공-->
            <div class="report-title-banner">
                <div style="float: left;">
                    <a style="font-size: 20px; font-weight: 900;">분석리포트</a>
                </div>
            </div>
            <div class="contents">
                <div class="contents" data-spy="scroll" data-target="#navbar-example2" data-offset="0">
                    <div id="result"></div>
                    <p id="info">부동산 매물 information</p>
                </div>
            </div>
        </div>
    </div>
      <script>
        function createDivBox() {
            var selectedValue = document.getElementById("inputState").value;
            if (selectedValue === "강남구") {
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="종로구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="중구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="용산구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="성동구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="광진구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="동대문구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="중랑구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="성북구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="강북구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="도봉구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="노원구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="은평구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="서대문구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="마포구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="양천구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="강서구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="구로구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="금천구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="영등포구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="동작구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="관악구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="서초구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="강동구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            } else if(selectedValue=="송파구"){
                document.querySelector('.result-box').style.display = 'block';
                document.querySelector('.report').style.display = 'block';
            }
        };
    
        document.querySelector('#okay1').addEventListener('click', function() {
            document.querySelector('.report').style.display = 'none';
            document.querySelector('.result-box').style.display = 'none';
            document.querySelector('.form-group').style.display = 'none';
            document.querySelector('.work-banner').style.display = 'block';
        });
    
        document.querySelector('#final').addEventListener('click',function() {
            document.querySelector('.report2').style.display = 'block';
        });
        
        const container = document.getElementById('map');
        const options = {
          center: new window.kakao.maps.LatLng(${centerLatLng.lat}, ${centerLatLng.lng}),
          level: 3
        };

          
        const map = new kakao.maps.Map(container, options);
                              
        kakao.maps.load(() => { 
          const markers = ${JSON.stringify(markers)};


          function addMarkers() {
            const clusterer = new kakao.maps.MarkerClusterer({
              map: map,
              averageCenter: true,
              minLevel:5,
              batchSize: 500,
              texts: getTexts
            });

            const bounds = map.getBounds();
            const positions = markers.filter(function(position) {
              return bounds.contain(new kakao.maps.LatLng(position.latlng.lat, position.latlng.lng));
            });
          
            let visibleMarkersCount;
            const zoomLevel = map.getLevel();
            if (zoomLevel <= 3) {
              visibleMarkersCount = Math.round(markers.length * 0.01); // 수정된 부분
            } else if (zoomLevel <= 4) {
              visibleMarkersCount = Math.round(markers.length * 0.05); // 수정된 부분
            } else if (zoomLevel <= 5) {
              visibleMarkersCount = Math.round(markers.length * 0.1); // 수정된 부분
            } else if (zoomLevel <= 6) {
                visibleMarkersCount = Math.round(markers.length * 0.15); // 수정된 부분
            } else if (zoomLevel <= 7) {
                visibleMarkersCount = Math.round(markers.length * 0.2); // 수정된 부분
            } else if (zoomLevel <= 8) {
                visibleMarkersCount = Math.round(markers.length * 0.3); // 수정된 부분
            } else if (zoomLevel <= 9) {
                visibleMarkersCount = Math.round(markers.length * 0.4); // 수정된 부분
            } else if (zoomLevel <= 10) {
                visibleMarkersCount = Math.round(markers.length * 0.5); // 수정된 부분
            } else {
              visibleMarkersCount = Math.round(markers.length * 1); // 수정된 부분
            }
            function getTexts(count){
              if(zoomLevel <= 3) {
                return Math.round(count*100);        
              } else if(zoomLevel <= 4) {
                return Math.round(count*20);
              } else if(zoomLevel <= 5) {
                return Math.round(count*10);
              } else if(zoomLevel <= 6) {
                return Math.round(count*5);
              } else if(zoomLevel <= 7) {
                return Math.round(count*20/3);
              } else {
                return count;
              }
            }
         
            const visibleMarkers = positions
              .sort(() => 0.5 - Math.random()) 
              .slice(0, visibleMarkersCount);
            
            
            const markersToAdd = visibleMarkers.map(function(position) {
              const marker = new kakao.maps.Marker({
                position: new kakao.maps.LatLng(position.latlng.lat, position.latlng.lng),
              });

              kakao.maps.event.addListener(marker, 'click', function(marker) { // 수정된 부분
                const link = "https://m.land.naver.com/article/info/" + position.atclNo;
                const clickedMarkerCoords = new kakao.maps.LatLng(position.latlng.lat, position.latlng.lng);
                const goods1=position.latlng.lat;
                const goods2=position.latlng.lng;
    
                someFunction(goods1,goods2);
                const markersWithSameCoords = markers.filter(function(marker) {
                  const markerCoords = new kakao.maps.LatLng(marker.latlng.lat, marker.latlng.lng);
                  return markerCoords.equals(clickedMarkerCoords);
              });
              flrInfo1=[];
              spc=[]
              const content = markersWithSameCoords.map(function(marker) {
                spc.push(marker.spc2);
                flrInfo1.push(marker.flrInfo);
                someFunction1(flrInfo1);
                someFunction2(spc);
                console.log(marker.spc2)
                return '<p>'+'<hr>' +'<'+ marker.atclFetrDesc +'>'+'<br>' + '건물 종류 : '+marker.atclNm +'<br>'+
                '임대 방식 : '+marker.tradTpNm + '<br>' + '건물 층 : '+marker.flrInfo + '<br>' +
                '월세 : ' + marker.prc+' 만원' + '<br>'+ '방향 : '+ marker.direction + '<br>'+'네이버 부동산 - '+
                '<a href="' + link + '">' + link + '</a></p>';
                }).join('');
                info.innerHTML = content;
              });
  
              return marker;
            });
            
          
            clusterer.clear();
            clusterer.addMarkers(markersToAdd);

          }

          function renderCircleOverlay(circle) {
              const radius = circle.getRadius();
              const center = circle.getPosition();

              const circleData = {
                  center: { lat: center.getLat(), lng: center.getLng() },
                  radius: radius
              };

              console.log(JSON.stringify(circleData));
          }
          
          kakao.maps.event.addListener(map, 'idle', function() {
            addMarkers();
          });

          
          const drawingOptions = { // Drawing Manager를 생성할 때 사용할 옵션입니다
            map: map, // Drawing Manager로 그리기 요소를 그릴 map 객체입니다
            drawingMode: [
              kakao.maps.drawing.OverlayType.CIRCLE,
            ],
            guideTooltip: ['draw', 'drag', 'edit'],
            circleOptions: {
              draggable: false,
              removable: true,
              editable: true,
              strokeColor: '#39f',
              fillColor: '#39f',
              fillOpacity: 0.5,
              maxRadius: 1000
            }
          };
    
          manager = new kakao.maps.drawing.DrawingManager(drawingOptions);
        });
    
        function selectOverlay(type) {
          // 그리기 중이면 그리기를 취소합니다
          manager.cancel();
    
          // 클릭한 그리기 요소 타입을 선택합니다
          manager.select(kakao.maps.drawing.OverlayType[type]);
        }
        const markers1 = ${JSON.stringify(markers)};
        function someFunction(goods1, goods2){
            goods10 = goods1;
            goods20 = goods2;
        }
        function someFunction1(flrInfo1){
            flrInfo10 = flrInfo1;
        }
        function someFunction2(spc){
            spc3 = spc;
            console.log(spc3)
        }
        function showCoordinatesAsJson() {
            const circles = manager.getData()[kakao.maps.drawing.OverlayType.CIRCLE];
            const markerCoordinates = [];
            let i = -1;
            circles.forEach(circle => {
              const circleCenter = circle.center;
              const circleRadius = circle.radius;
              const markersInCircle = markers1.filter(marker => {
                const markerPosition = marker.latlng;
                const distance = Math.sqrt(
                  Math.pow(markerPosition.lat - circleCenter.y, 2) +
                  Math.pow(markerPosition.lng - circleCenter.x, 2)
                );
                return goods10 == markerPosition.lat && goods20 === markerPosition.lng;
              });
          
              const coordinatesInCircle = markersInCircle.map(marker => {
                i++;
                return {
                  circleX: circleCenter.x,
                  circleY: circleCenter.y,
                  flrInfo: flrInfo10[i],
                  lng: goods10,
                  lat: goods20,
                  spc: spc3[i]
                };
              });
              markerCoordinates.push(...coordinatesInCircle);
            });
          
            const json = JSON.stringify(markerCoordinates);
            console.log(json)
            
            // POST 요청 보내기
      fetch('/researchhome', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(markerCoordinates),
      })
        .then(response => response.text())
        .then(result => {
          // 분석 결과를 받아서 웹에 표시
          const resultContainer = document.getElementById('result');
          resultContainer.innerText = result;
        })
        .catch(error => {
          console.error('에러 발생:', error);
        });
          }
          
          
      </script>
    </body>
  </html> `;
  }
  module.exports = researchhome;