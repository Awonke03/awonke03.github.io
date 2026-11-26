import os,sys,subprocess,time
from playwright.sync_api import sync_playwright,expect
server=subprocess.Popen([sys.executable,'-m','http.server','8765'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
try:
    time.sleep(1)
    with sync_playwright() as p:
        b=p.chromium.launch()
        page=b.new_page(viewport={'width':1440,'height':900})
        errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto('http://127.0.0.1:8765/sentiscope/')
        expect(page.locator('#status')).to_contain_text('30 comments',timeout=60000)
        expect(page.locator('#donut svg')).to_be_visible()
        page.locator('[data-view="text"]').click()
        page.locator('#comment').fill('The app is excellent but the service is terrible.')
        page.locator('#analyse').click()
        expect(page.locator('#text-status')).to_have_text('Analysis complete')
        expect(page.locator('#aspects .result')).to_have_count(2)
        page.locator('#file').set_input_files({'name':'feedback.csv','mimeType':'text/csv','buffer':b'text,language\nGreat service,en\nBad payment,en\n'})
        expect(page.locator('#status')).to_contain_text('2 comments')
        page.locator('[data-view="feedback"]').click()
        expect(page.locator('#records tbody tr')).to_have_count(2)
        page.locator('[data-view="search"]').click()
        page.locator('#search-mode').select_option('lexical')
        page.locator('#query').fill('payment')
        page.locator('#find').click()
        expect(page.locator('#matches .result')).to_have_count(2)
        for width in [320,390,768,1024,1440]:
            page.set_viewport_size({'width':width,'height':900})
            for v in ['overview','text','feedback','topics','search']:
                page.locator('[data-view="'+v+'"]').click()
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),(width,v)
        assert not errors,errors
        b.close()
finally:server.terminate()
