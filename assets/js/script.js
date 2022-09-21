(function () {
  const menu = document.querySelectorAll('#menu--main a')
  const checkbox = document.querySelector('#nav__toggle--main')

  if (!menu || !menu.length || !checkbox) {
    return
  }
  
  for (let i = 0; i < menu.length; i++) {
    menu[i].addEventListener('click', function (event) {
      console.log('%cClicked: ', 'color: orange;', event)
      checkbox.checked = false
    })
  }
})()
